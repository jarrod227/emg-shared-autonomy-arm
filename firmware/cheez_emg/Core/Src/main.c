/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "dma.h"
#include "tim.h"
#include "usb_device.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdbool.h>

#include "emg_activation.h"
#include "emg_view.h"
#include "emg_classifier.h"
#include "emg_features.h"
#include "emg_filter.h"
#include "emg_gate.h"
#include "emg_packet.h"
#include "emg_rx.h"
#include "usbd_cdc_if.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define EMG_CHANNELS 3u
#define EMG_FRAMES_PER_PACKET 32u
#define EMG_SAMPLE_RATE_HZ 2000u
#define EMG_ADC_BITS 12u
#define EMG_FIRMWARE_VERSION 0x0100u

/* One half-buffer is exactly one RAW packet, so a half-transfer interrupt
 * produces one packet with nothing left over. */
#define EMG_HALF_SAMPLES (EMG_FRAMES_PER_PACKET * EMG_CHANNELS)
#define EMG_FRAME_PERIOD_US (1000000u / EMG_SAMPLE_RATE_HZ)
#define EMG_TX_BUFFER_SIZE (EMG_HEADER_SIZE + EMG_RAW_HEADER_SIZE \
                            + EMG_HALF_SAMPLES * 2u + EMG_CRC_SIZE)
#define EMG_INFO_PERIOD_MS 2000u
#define EMG_TX_TIMEOUT_MS 5u

/* Must equal ZERO_CROSSING_THRESHOLD in firmware/tools/emg_train_lda.py. The
 * model was fitted on features counted with this threshold; a different one
 * here changes one of the twelve inputs and silently invalidates it. */
#define EMG_ZERO_CROSSING_THRESHOLD 10u
#define EMG_WEAR_ALL_ATTACHED ((1u << EMG_CHANNELS) - 1u)

/* confidence = min(255, (top score - runner-up) >> this). Diagnostic only:
 * the mapping is reconstructible by the host but no threshold on it has been
 * validated, so nothing may gate on it. See PROTOCOL.md. */
#define EMG_CONFIDENCE_SHIFT 16
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
/* Circular DMA target, two halves. No lock guards it: by the time an
 * interrupt announces a half, the DMA is already writing the other one, so
 * the CPU never reads the half being written. */
static uint16_t emg_adc_buffer[2u * EMG_HALF_SAMPLES];

/* Written only by the ADC interrupt, read only by the main loop. A single
 * counter is enough -- which physical half is ready follows from its parity. */
static volatile uint32_t emg_halves_produced;
static uint32_t emg_halves_consumed;

/* CDC_Transmit_FS hands the pointer to the USB driver and returns while the
 * transfer is still in flight, so encoding the next packet into the same
 * memory would corrupt the one being sent. Two buffers, used alternately. */
static uint8_t emg_tx_buffer[2][EMG_TX_BUFFER_SIZE];
static uint8_t emg_tx_slot;

static uint16_t emg_info_sequence;
static uint32_t emg_next_info_tick;

static emg_filter_t emg_filters[EMG_CHANNELS];
static emg_feature_window_t emg_windows[EMG_CHANNELS];
static uint32_t emg_saturations[EMG_CHANNELS];
static emg_gate_t emg_gate;
static emg_activation_t emg_activation;
static uint16_t emg_intent_sequence;

/* Host-to-device receive path. Not static: the USB CDC receive callback in
 * usbd_cdc_if.c pushes into it from interrupt context. Zero-initialised is
 * a valid empty state, so bytes arriving before USER CODE 2 runs land in a
 * working ring rather than undefined behaviour. */
emg_rx_t emg_usb_rx;

/* How the activation threshold came to hold its current values, reported in
 * every ACTIVATION_STATE packet. RAM only: a reset returns to the compile-
 * time defaults and the host re-sends on its startup handshake. */
static uint8_t emg_activation_source; /* emg_activation_source_t */
static uint8_t emg_activation_last_result; /* emg_set_result_t */
static uint16_t emg_activation_applied_sequence;
static uint16_t emg_state_sequence;

/* Frames processed by the DSP, which is not the same as frames captured: the
 * count only advances over samples that actually entered the filters, so the
 * INTENT timestamp names the frame the decision was made on and the host can
 * align it against the RAW stream exactly. */
static uint32_t emg_frames_processed;

/* Saturates at the window length. A feature window is usable only when every
 * frame in it was contiguous and fully attached, which is the same rule the
 * host applies, so this counter is compared against EMG_FEATURES_WINDOW rather
 * than tracking per-frame validity. */
static uint16_t emg_frames_since_invalid;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
static uint8_t emg_read_wear_mask(void);
static bool emg_transmit(const uint8_t *data, uint16_t length);
static void emg_send_info(void);
static void emg_send_activation_state(void);
static void emg_apply_set_activation(const emg_set_activation_t *request);
static void emg_send_raw_half(uint32_t index, uint8_t wear_mask);
static void emg_dsp_init(void);
static void emg_dsp_discontinuity(void);
static void emg_process_half(uint32_t index, bool attached);
static void emg_send_intent(emg_command_t command,
                            const emg_classification_t *result,
                            bool window_valid, uint32_t new_saturations,
                            emg_command_t decision, int32_t total_mav);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static uint8_t emg_read_wear_mask(void)
{
  uint8_t mask = 0u;

  /* Sensor column A0 -> PA8, A1 -> PA2, A2 -> PA3; see firmware/README.md.
   * The pins are pulled down, so an unplugged column reads as no contact --
   * the fail-closed direction. Polarity is assumed active-high and has not
   * been verified against the module yet. */
  if (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_8) == GPIO_PIN_SET) { mask |= 1u << 0; }
  if (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_2) == GPIO_PIN_SET) { mask |= 1u << 1; }
  if (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_3) == GPIO_PIN_SET) { mask |= 1u << 2; }
  return mask;
}

static bool emg_transmit(const uint8_t *data, uint16_t length)
{
  const uint32_t start = HAL_GetTick();

  /* Subtraction rather than a computed deadline, so a tick wrap cannot make
   * the timeout fire immediately or never. */
  while (CDC_Transmit_FS((uint8_t *)data, length) == USBD_BUSY) {
    if ((HAL_GetTick() - start) >= EMG_TX_TIMEOUT_MS) {
      return false;
    }
  }
  return true;
}

static void emg_send_info(void)
{
  const emg_info_t info = {
      EMG_FIRMWARE_VERSION, EMG_SAMPLE_RATE_HZ, (uint8_t)EMG_CHANNELS,
      (uint8_t)EMG_ADC_BITS, (uint8_t)EMG_FRAMES_PER_PACKET,
  };
  uint8_t *buffer = emg_tx_buffer[emg_tx_slot];
  const size_t length = emg_encode_info(
      buffer, EMG_TX_BUFFER_SIZE, emg_info_sequence,
      emg_halves_consumed * EMG_FRAMES_PER_PACKET * EMG_FRAME_PERIOD_US, &info);

  if (length != 0u) {
    emg_tx_slot ^= 1u;
    (void)emg_transmit(buffer, (uint16_t)length);
  }
  /* Advance even when the send failed: the host should see a gap rather than
   * a repeat, because a repeat looks like working firmware. */
  emg_info_sequence++;
}

static void emg_send_activation_state(void)
{
  emg_activation_state_t state;
  uint8_t *buffer = emg_tx_buffer[emg_tx_slot];
  size_t length;

  state.source = emg_activation_source;
  state.factor = (uint8_t)emg_activation.factor;
  state.baseline_shift = (uint8_t)emg_activation.baseline_shift;
  state.last_result = emg_activation_last_result;
  state.threshold_floor = emg_activation.threshold_floor;
  state.applied_sequence = emg_activation_applied_sequence;
  length = emg_encode_activation_state(
      buffer, EMG_TX_BUFFER_SIZE, emg_state_sequence,
      emg_frames_processed * EMG_FRAME_PERIOD_US, &state);
  if (length != 0u)
  {
    emg_tx_slot ^= 1u;
    (void)emg_transmit(buffer, (uint16_t)length);
  }
  emg_state_sequence++;
}

/* Apply one decoded host request and report the outcome immediately, so the
 * sender never waits a full periodic-state interval to learn it. A rejected
 * request changes nothing: emg_activation_reconfigure validates before it
 * touches state, so there is no partial apply to undo. */
static void emg_apply_set_activation(const emg_set_activation_t *request)
{
  bool accepted = false;

  if (request->mode == (uint8_t)EMG_SET_MODE_DEFAULTS)
  {
    /* A deliberate un-calibration: a stale calibration from a previous
     * wearer or donning is worse than none. Value fields are ignored. */
    accepted = emg_activation_reconfigure(&emg_activation,
                                          EMG_ACTIVATION_FACTOR,
                                          EMG_ACTIVATION_BASELINE_SHIFT,
                                          EMG_ACTIVATION_THRESHOLD_FLOOR);
    if (accepted)
    {
      emg_activation_source = (uint8_t)EMG_ACTIVATION_SOURCE_DEFAULTS;
    }
  }
  else if (request->mode == (uint8_t)EMG_SET_MODE_APPLY)
  {
    accepted = emg_activation_reconfigure(&emg_activation, request->factor,
                                          request->baseline_shift,
                                          request->threshold_floor);
    if (accepted)
    {
      emg_activation_source = (uint8_t)EMG_ACTIVATION_SOURCE_HOST;
    }
  }

  if (accepted)
  {
    emg_activation_last_result = (uint8_t)EMG_SET_RESULT_ACCEPTED;
    emg_activation_applied_sequence = request->sequence;
  }
  else
  {
    emg_activation_last_result = (uint8_t)EMG_SET_RESULT_REJECTED;
  }
  emg_send_activation_state();
}

static void emg_send_raw_half(uint32_t index, uint8_t wear_mask)
{
  const uint16_t *half = &emg_adc_buffer[(index % 2u) * EMG_HALF_SAMPLES];
  uint8_t *buffer = emg_tx_buffer[emg_tx_slot];

  /* The sample clock is hardware-timed, so the timestamp is derived from the
   * frame count rather than from HAL_GetTick, whose 1 ms resolution is
   * coarser than a 0.5 ms sample period. Wraps every 71.6 minutes, which is
   * what PROTOCOL.md specifies. */
  const size_t length = emg_encode_raw(
      buffer, EMG_TX_BUFFER_SIZE, (uint16_t)index,
      index * EMG_FRAMES_PER_PACKET * EMG_FRAME_PERIOD_US,
      wear_mask, half, (uint16_t)EMG_HALF_SAMPLES);

  if (length != 0u) {
    emg_tx_slot ^= 1u;
    (void)emg_transmit(buffer, (uint16_t)length);
  }
}

static void emg_dsp_init(void)
{
  emg_gate_config_t gate_config;

  for (uint32_t channel = 0u; channel < EMG_CHANNELS; channel++) {
    if (!emg_filter_init(&emg_filters[channel],
                         emg_filter_20_450_notch60_at_2000,
                         EMG_FILTER_DEFAULT_SECTIONS)) {
      Error_Handler();
    }
    if (!emg_features_init(&emg_windows[channel],
                           EMG_ZERO_CROSSING_THRESHOLD)) {
      Error_Handler();
    }
    emg_saturations[channel] = 0u;
  }
  emg_gate_default_config(&gate_config);
  if (!emg_gate_init(&emg_gate, &gate_config)) {
    Error_Handler();
  }
  if (!emg_activation_init(&emg_activation, EMG_ACTIVATION_FACTOR,
                           EMG_ACTIVATION_BASELINE_SHIFT,
                           EMG_ACTIVATION_THRESHOLD_FLOOR)) {
    Error_Handler();
  }
  emg_frames_since_invalid = 0u;
}

/* A gap in the sample stream is not a small error. The biquad state and the
 * feature window both assume contiguous samples, so carrying them across a gap
 * produces plausible features from a signal that never existed. Everything is
 * reset and the gate is told the evidence is gone. */
static void emg_dsp_discontinuity(void)
{
  for (uint32_t channel = 0u; channel < EMG_CHANNELS; channel++) {
    emg_filter_reset(&emg_filters[channel]);
    emg_features_reset(&emg_windows[channel]);
    emg_saturations[channel] = emg_windows[channel].saturations;
  }
  emg_gate_invalidate(&emg_gate);
  /* The activation baseline is deliberately left alone: rest amplitude is a
   * property of the wearer and the donning, not of stream continuity, so a
   * dropout should not strip the amplitude protection that survives it. */
  emg_frames_since_invalid = 0u;
}

/* `command` is the event gate's output and `decision` the post-activation
 * per-hop one. They are different readings and both are needed: the discrete
 * half of the packet carries the event, the proportional half carries what
 * the muscle is doing right now. */
static void emg_send_intent(emg_command_t command,
                            const emg_classification_t *result,
                            bool window_valid, uint32_t new_saturations,
                            emg_command_t decision, int32_t total_mav)
{
  uint8_t *buffer = emg_tx_buffer[emg_tx_slot];
  int64_t best = result->scores[0];
  int64_t runner_up = INT64_MIN;
  emg_intent_t intent;

  for (uint32_t class_index = 1u;
       class_index < EMG_CLASSIFIER_CLASS_COUNT;
       class_index++) {
    const int64_t score = result->scores[class_index];
    if (score > best) {
      runner_up = best;
      best = score;
    } else if (score > runner_up) {
      runner_up = score;
    }
  }

  const int64_t margin = (best - runner_up) >> EMG_CONFIDENCE_SHIFT;
  intent.command = (uint8_t)command;
  intent.confidence = (margin >= 255) ? 255u : (uint8_t)margin;
  /* Quality is contact and headroom, not certainty: a clipped or detached
   * window can still produce a confident wrong score. */
  if (!window_valid) {
    intent.signal_quality = 0u;
  } else if (new_saturations >= 255u) {
    intent.signal_quality = 0u;
  } else {
    intent.signal_quality = (uint8_t)(255u - new_saturations);
  }
  /* Proportional view control is not implemented. */
  /* The proportional half of the same decision. Derived from the
   * post-activation decision, not the classifier output: a window too quiet
   * to be an intent is also too quiet to steer with. The threshold is
   * recomputed rather than stored so it can never disagree with what the
   * activation stage just applied. */
  {
    const int32_t baseline = emg_activation_baseline(&emg_activation);
    const int32_t relative = (int32_t)emg_activation.factor * baseline;
    const int32_t threshold = relative > emg_activation.threshold_floor
                                  ? relative
                                  : emg_activation.threshold_floor;
    intent.direction = emg_view_direction(decision);
    intent.activation = emg_view_activation(total_mav, threshold);
  }

  const size_t length = emg_encode_intent(
      buffer, EMG_TX_BUFFER_SIZE, emg_intent_sequence,
      emg_frames_processed * EMG_FRAME_PERIOD_US, &intent);
  if (length != 0u) {
    emg_tx_slot ^= 1u;
    (void)emg_transmit(buffer, (uint16_t)length);
  }
  emg_intent_sequence++;
}

static void emg_process_half(uint32_t index, bool attached)
{
  const uint16_t *half = &emg_adc_buffer[(index % 2u) * EMG_HALF_SAMPLES];

  for (uint32_t frame = 0u; frame < EMG_FRAMES_PER_PACKET; frame++) {
    emg_features_t features[EMG_CHANNELS] = {0};
    uint32_t hops = 0u;

    /* One wear reading covers the whole half, matching what the host sees:
     * a RAW packet carries a single mask that it applies to every frame in
     * the packet. Reading per frame here would make the two disagree. */
    if (!attached) {
      emg_frames_since_invalid = 0u;
    } else if (emg_frames_since_invalid < EMG_FEATURES_WINDOW) {
      emg_frames_since_invalid++;
    }

    for (uint32_t channel = 0u; channel < EMG_CHANNELS; channel++) {
      /* 12-bit unsigned counts fit int16 as they are. The band-pass removes
       * the offset, and the host feeds the filter the same raw counts, so no
       * centring is applied on either side. */
      const int16_t raw = (int16_t)half[frame * EMG_CHANNELS + channel];
      const int32_t filtered = emg_filter_step(&emg_filters[channel], raw);
      if (emg_features_push(&emg_windows[channel], filtered,
                            &features[channel])) {
        hops++;
      }
    }
    emg_frames_processed++;

    /* Every channel is pushed for every frame, so the three cross a hop on the
     * same sample. Requiring all three rather than testing one means a future
     * edit that lets them drift apart produces no classification instead of a
     * classification built from one stale channel. */
    if (hops == EMG_CHANNELS) {
      emg_classification_t result;
      emg_command_t event = EMG_COMMAND_REST;
      const bool window_valid =
          (emg_frames_since_invalid >= EMG_FEATURES_WINDOW);
      uint32_t new_saturations = 0u;

      for (uint32_t channel = 0u; channel < EMG_CHANNELS; channel++) {
        new_saturations +=
            emg_windows[channel].saturations - emg_saturations[channel];
        emg_saturations[channel] = emg_windows[channel].saturations;
      }
      if (!emg_classifier_predict(features, &result)) {
        continue;
      }
      /* Shape first, then strength: the classifier names the gesture the
       * window resembles, and the activation stage decides whether enough
       * muscle was behind it to count as an intent at all. A preparatory
       * movement at twice resting amplitude classifies correctly as a
       * gesture and is still not one. */
      const int32_t total_mav = features[0].mean_absolute_value
                                + features[1].mean_absolute_value
                                + features[2].mean_absolute_value;
      const emg_command_t decision = emg_activation_apply(
          &emg_activation, result.command, window_valid, total_mav);
      /* REST is reported too. A silent link is indistinguishable from a dead
       * one, so the absence of intent is stated rather than implied; the
       * command carries the gate's event, so anything other than REST means
       * an event fired on this hop. */
      (void)emg_gate_push(&emg_gate, decision, window_valid, &event);
      emg_send_intent(event, &result, window_valid, new_saturations,
                      decision, total_mav);
    }
  }
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USB_DEVICE_Init();
  MX_ADC1_Init();
  MX_TIM3_Init();
  /* USER CODE BEGIN 2 */
  /* Before the ADC starts, so no half can arrive at an uninitialised filter. */
  emg_dsp_init();
  /* The F1 ADC needs a self-calibration after power-up. Skipping it leaves a
   * few counts of offset that nothing downstream can recover. */
  if (HAL_ADCEx_Calibration_Start(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_ADC_Start_DMA(&hadc1, (uint32_t *)emg_adc_buffer,
                        (uint32_t)(2u * EMG_HALF_SAMPLES)) != HAL_OK)
  {
    Error_Handler();
  }
  /* Timer last: the ADC has to be armed before the first trigger arrives. */
  if (HAL_TIM_Base_Start(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  emg_next_info_tick = HAL_GetTick();
/* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    if ((int32_t)(HAL_GetTick() - emg_next_info_tick) >= 0)
    {
      emg_send_info();
      /* On the INFO cadence so a host that just attached learns what the
       * board is judging with without asking. */
      emg_send_activation_state();
      emg_next_info_tick += EMG_INFO_PERIOD_MS;
    }

    {
      /* One request per loop pass keeps arrival order and bounds the time
       * spent here; configuration traffic is a few packets per session. */
      emg_set_activation_t request;
      if (emg_rx_poll(&emg_usb_rx, &request))
      {
        emg_apply_set_activation(&request);
      }
    }

    const uint32_t produced = emg_halves_produced;
    if (produced != emg_halves_consumed)
    {
      /* More than one half outstanding means the DMA has already wrapped past
       * the oldest one and is overwriting it, so sending it would ship torn
       * data. Skip to the newest complete half instead; the sequence gap
       * reports the loss to the host through the same path a dropped USB
       * packet does. */
      if ((produced - emg_halves_consumed) > 1u)
      {
        emg_halves_consumed = produced - 1u;
        /* The RAW stream survives a gap; the DSP does not, so it is told. */
        emg_dsp_discontinuity();
      }
      /* Read once and share it. RAW reports the mask to the host and the DSP
       * decides window validity from it, so two reads could disagree and make
       * the host's replay diverge from the firmware's own decision. */
      const uint8_t wear_mask = emg_read_wear_mask();
      emg_send_raw_half(emg_halves_consumed, wear_mask);
      emg_process_half(emg_halves_consumed, wear_mask == EMG_WEAR_ALL_ATTACHED);
      emg_halves_consumed++;
    }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_ADC|RCC_PERIPHCLK_USB;
  PeriphClkInit.AdcClockSelection = RCC_ADCPCLK2_DIV6;
  PeriphClkInit.UsbClockSelection = RCC_USBCLKSOURCE_PLL_DIV1_5;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
/* Both callbacks deliberately do almost nothing. All the work is left to the
 * main loop so that a busy USB endpoint can never stall an interrupt, and so
 * the interrupt cannot run long enough to overlap the next one. */
void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
  if (hadc->Instance == ADC1)
  {
    emg_halves_produced++;
  }
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
  if (hadc->Instance == ADC1)
  {
    emg_halves_produced++;
  }
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
