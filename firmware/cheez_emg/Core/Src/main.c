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

#include "emg_packet.h"
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
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
static uint8_t emg_read_wear_mask(void);
static bool emg_transmit(const uint8_t *data, uint16_t length);
static void emg_send_info(void);
static void emg_send_raw_half(uint32_t index);
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

static void emg_send_raw_half(uint32_t index)
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
      emg_read_wear_mask(), half, (uint16_t)EMG_HALF_SAMPLES);

  if (length != 0u) {
    emg_tx_slot ^= 1u;
    (void)emg_transmit(buffer, (uint16_t)length);
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
      emg_next_info_tick += EMG_INFO_PERIOD_MS;
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
      }
      emg_send_raw_half(emg_halves_consumed);
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
