# Literature Ledger

Last reviewed: 2026-07-26

This ledger controls the boundary between background reading and claims that
the project may cite. It is not evidence that the proposed system is novel.
Novelty must be argued against a broader, systematic search and the measured
project contribution.

## Promotion workflow

The workflow has two controlled levels:

1. **Ledger level — candidate to verified**
   - `CANDIDATE`: relevant source found, but the primary text and metadata have
     not both been checked. It must not support project prose as fact and must
     not appear in `references.bib`.
   - `VERIFIED`: the primary paper or official documentation was checked; the
     exact supported claim and its limits are recorded below.
2. **Citation level — verified to formal**
   - `FORMAL`: a verified source is actually used in a named project section
     and has been promoted to `references.bib`.
   - Removal from project prose also removes the need for a formal entry;
     `references.bib` is not a general reading list.

Promotion to `VERIFIED` requires title, authors or responsible organization,
year/version where available, primary URL, and DOI when one exists. Promotion
to `FORMAL` additionally requires a unique cite key and a specific use
location. Ambiguous fields are omitted rather than inferred.

Research papers and official software documentation are tracked separately.
Software documentation may establish API behavior, message semantics, or
implementation constraints; it does not establish research novelty or
experimental performance.

## Formally cited research papers

All entries in this section are `FORMAL` and have matching entries in
`references.bib`.

### `zhang2000flexible`

- Type: research paper
- Primary source: [Microsoft Research publication page](https://www.microsoft.com/en-us/research/?p=145334)
- DOI: [10.1109/34.888718](https://doi.org/10.1109/34.888718)
- Verified claim: a camera can be calibrated from multiple views of a planar
  pattern using a closed-form initialization followed by nonlinear refinement,
  including radial-distortion modeling.
- Claim boundary: it does not validate this project's stereo extrinsics,
  timestamp synchronization, hand-eye transform, or working-range error.
- Use: `docs/proposal.md` §4 research boundary and Objective 3.2/5 calibration
  plan; `docs/system_design.md` “Stereo Sensing Foundation.”

### `hirschmuller2008stereo`

- Type: research paper
- Primary source: [DLR publication record and author manuscript](https://elib.dlr.de/55367/)
- DOI: [10.1109/TPAMI.2007.1166](https://doi.org/10.1109/TPAMI.2007.1166)
- Verified claim: Semi-Global Matching approximates a global stereo objective
  through pathwise cost aggregation and includes disparity refinement and
  outlier handling.
- Claim boundary: it does not prove that OpenCV StereoSGBM parameters are
  suitable for the purchased cameras, nor that approximately paired frames are
  exposure-synchronized.
- Use: `docs/proposal.md` §4 and Objective 3.2; `docs/system_design.md`
  “Stereo Sensing Foundation.”

### `he2017maskrcnn`

- Type: research paper
- Primary source: [ICCV open-access proceedings](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)
- DOI: [10.1109/ICCV.2017.322](https://doi.org/10.1109/ICCV.2017.322)
- Verified claim: Mask R-CNN extends object detection with a parallel,
  per-instance mask branch and is an established instance-segmentation
  architecture.
- Claim boundary: it does not select the final lightweight model, establish
  edge-device feasibility, or prove accuracy on this project's objects.
- Use: `docs/proposal.md` §4 and Objective 3.2; `docs/system_design.md`
  “Perception and Selection.”

### `zhang2020mediapipehands`

- Type: research paper
- Primary source: [arXiv primary record](https://arxiv.org/abs/2006.10214)
- Verified claim: MediaPipe Hands describes a real-time, on-device hand
  tracking pipeline with a palm detector followed by a hand-landmark model for
  monocular RGB input.
- Claim boundary: its reported landmarks are not a calibrated stereo 3D hand
  observation and are not a safety-rated human-presence signal.
- Use: `docs/proposal.md` §4 and Objective 4.2; `docs/system_design.md`
  hand-observation design.

### `lopez2009robustemg`

- Type: research paper
- Primary source: [PubMed record](https://pubmed.ncbi.nlm.nih.gov/19243627/)
- Full text: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC2657216/)
- DOI: [10.1186/1475-925X-8-5](https://doi.org/10.1186/1475-925X-8-5)
- Verified claim: EMG amplitude was mapped proportionally to a robot-arm joint,
  and redundant EMG fusion plus real-time DSP implementation was evaluated for
  robustness to electrode and noise faults.
- Claim boundary: it does not validate this project's three-channel STM32
  classifier, activation calibration, command mapping, or safety response.
- Use: `docs/proposal.md` §4 and Objective 3.5; `docs/system_design.md`
  “Embedded EMG Intent.”

### `minati2016hybrid`

- Type: research paper
- Primary source: [IEEE Xplore record](https://ieeexplore.ieee.org/document/7805178/)
- DOI: [10.1109/ACCESS.2017.2647851](https://doi.org/10.1109/ACCESS.2017.2647851)
- Verified claim: a close prior system combined consumer-grade EOG, EMG, EEG,
  and head movement with a vision-guided robot arm.
- Claim boundary: the paper establishes related system integration, not the
  proposed bounded EMG intervention experiment, STM32 implementation, or this
  project's stereo accuracy and safety contracts.
- Use: `docs/proposal.md` §4 research novelty boundary.
- Metadata note: the archival IEEE Access volume is 4 (2016); the DOI string
  contains 2017 and the landing page appeared later, but that is not used to
  change the volume year.

### `ross2011dagger`

- Type: research paper
- Primary source: [PMLR proceedings](https://proceedings.mlr.press/v15/ross11a.html)
- Verified claim: DAgger iteratively aggregates expert labels on states induced
  by the learned policy and frames the procedure as a no-regret reduction for
  sequential prediction.
- Claim boundary: ordinary DAgger does not model a low-bandwidth EMG channel,
  human takeover cost, abort latency, or action chunks.
- Use: `docs/proposal.md` Phase 3 Objective 6; `TODO.md` P3.2.

### `kelly2019hgdagger`

- Type: research paper
- Primary source: [arXiv author manuscript](https://arxiv.org/abs/1810.02890)
- DOI: [10.1109/ICRA.2019.8793698](https://doi.org/10.1109/ICRA.2019.8793698)
- Verified claim: HG-DAgger adapts data aggregation to human-gated
  intervention and learns an uncertainty-based risk threshold.
- Claim boundary: it evaluates human intervention in autonomous driving, not
  EMG-gated manipulation or action-chunk abort latency.
- Use: `docs/proposal.md` Phase 3 novelty boundary; `TODO.md` P3.2.

### `hoque2022thriftydagger`

- Type: research paper
- Primary source: [PMLR proceedings](https://proceedings.mlr.press/v164/hoque22a.html)
- Verified claim: ThriftyDAgger uses novelty and risk gating to request human
  intervention under an intervention budget and explicitly evaluates task
  performance against supervisor burden.
- Claim boundary: it does not study EMG decoding, proportional view control,
  or ACT chunk-length versus abort latency.
- Use: `docs/proposal.md` §4 and Phase 3 Objective 6; `TODO.md` P3.2.

### `zhao2023act`

- Type: research paper
- Primary source: [Robotics: Science and Systems proceedings](https://roboticsproceedings.org/rss19/p016.html)
- DOI: [10.15607/RSS.2023.XIX.016](https://doi.org/10.15607/RSS.2023.XIX.016)
- Verified claim: Action Chunking with Transformers predicts sequences of
  actions in chunks to reduce the effective horizon for end-to-end imitation
  learning on fine-grained bimanual manipulation.
- Claim boundary: ACT does not establish that a particular chunk length is
  optimal, nor does it evaluate EMG abort latency; that interaction remains a
  candidate project experiment.
- Use: `docs/proposal.md` §4 and Phase 3 Objective 6; `docs/system_design.md`
  learned backend; `TODO.md` P3.2.

### `liu2025sirius`

- Type: research paper
- Primary source: [International Journal of Robotics Research](https://journals.sagepub.com/doi/10.1177/02783649241273901)
- DOI: [10.1177/02783649241273901](https://doi.org/10.1177/02783649241273901)
- Verified claim: Sirius studies human intervention during robot deployment and
  reweights collected samples using approximated human trust for subsequent
  behavioral-cloning updates.
- Claim boundary: Sirius does not study biosignal-channel constraints or ACT
  chunk-length versus abort latency.
- Use: `docs/proposal.md` §4 and Phase 3 Objective 6; `TODO.md` P3.2.
- Metadata note: first published online in 2024; the formal journal issue is
  volume 44, issues 10–11 (2025), pages 1727–1742, which is used in BibTeX.

## Verified official software documentation

These entries are `VERIFIED`, but are not currently promoted to
`references.bib`. Promote one only when a report explicitly relies on its API
or message-semantics claim.

### `DOC-OPENCV-CALIB3D`

- Responsible organization: OpenCV
- Official page: [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html)
- Verified claim: OpenCV provides camera/stereo calibration, rectification,
  triangulation, and disparity-to-3D APIs; points reprojected with the `Q`
  matrix from `stereoRectify` are expressed in the first rectified camera
  frame.
- Claim boundary: API availability is not an accuracy guarantee.
- Intended use: Objective 4.2/3.2 implementation notes and calibration
  verification plan.

### `DOC-OPENCV-STEREOSGBM`

- Responsible organization: OpenCV
- Official page: [`cv::StereoSGBM` reference](https://docs.opencv.org/4.10.0/d2/d85/classcv_1_1StereoSGBM.html)
- Verified claim: OpenCV exposes an SGBM-family dense stereo matcher and its
  parameter semantics.
- Claim boundary: the implementation differs in details from the complete
  Hirschmüller formulation; documentation does not establish suitable
  parameters or project performance.
- Intended use: Objective 3.2 implementation notes.

### `DOC-ROS2-APPROX-TIME`

- Responsible organization: Open Robotics / ROS 2
- Official page: [ROS 2 Jazzy Approximate Time tutorial](https://docs.ros.org/en/ros2_packages/jazzy/api/message_filters/doc/Tutorials/Approximate-Synchronizer-Cpp.html)
- Verified claim: `ApproximateTime` aligns messages using timestamps and a
  queue/policy, and matching QoS is required for the documented setup.
- Claim boundary: message pairing does not trigger simultaneous camera
  exposures and therefore is not hardware synchronization.
- Intended use: `docs/system_design.md` stereo-pairing contract and Objective
  4.2/3.2 implementation.

### `DOC-MEDIAPIPE-HAND-LANDMARKER`

- Responsible organization: Google
- Official page: [MediaPipe Hand Landmarker Python API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker)
- Verified claim: the API supports image, video, and asynchronous live-stream
  inference; live-stream results are delivered through a callback and frames
  may be dropped to reduce latency.
- Claim boundary: the API does not provide this project's stereo
  correspondence, metric triangulation, temporal-stability gate, or safety
  certification.
- Intended use: Objective 4.2 implementation notes.

### `DOC-ST-RM0008`

- Responsible organization: STMicroelectronics
- Official page: [RM0008 — STM32F101xx/102xx/103xx/105xx/107xx reference
  manual](https://www.st.com/resource/en/reference_manual/rm0008-stm32f101xx-stm32f102xx-stm32f103xx-stm32f105xx-and-stm32f107xx-advanced-armbased-32bit-mcus-stmicroelectronics.pdf)
- Version checked: Rev 21 (local copy `docs/ReferenceManual.pdf`)
- Verified claim: on this MCU family only ADC1 and ADC3 have DMA capability —
  "Only ADC1 and ADC3 have this DMA capability. ADC2-converted data can be
  transferred in dual ADC mode using DMA thanks to master ADC1" (p. 227).
  Table 78 lists ADC1 on DMA1 channel 1 and contains no ADC2 entry at all.
  DMA2 "and its relative requests are available only in high-density,
  XL-density and connectivity line devices", so the medium-density
  STM32F103C8T6 has neither ADC3 nor DMA2. Scan mode is described as
  "automatic conversion of channel 0 to channel 'n'", i.e. sequential.
  Conversion time is "1 µs at 56 MHz (1.17 µs at 72 MHz)" and the clock tree
  caps ADCCLK at 14 MHz.
- Claim boundary: this fixes what the silicon can do; it does not establish
  the sampling rate, inter-channel skew, or dropped-sample behaviour of any
  firmware actually written for this project. Those remain to be measured.
  It also says nothing about the purchased Cheez sEMG board's analog front
  end or channel wiring.
- Intended use: Objective 3.5 firmware design — the choice of ADC1 scan mode
  + DMA1 channel 1 for three channels, and the corrected wording of the
  three-channel acquisition item in `TODO.md`.

## Verified research papers not yet formally cited

`VERIFIED` under the promotion workflow: the primary text and metadata were
both checked, and the entry may support project prose. It is not yet `FORMAL`
because no named project section cites it, so it has no `references.bib`
entry. Promote it on first use.

### `zhu2020emgforce`

- Type: research paper
- Primary source: [IEEE Xplore record](https://ieeexplore.ieee.org/document/9260149)
- Full text: [author's final copy (WPI)](https://users.wpi.edu/~ted/full_text/2020zhu_martinezluna_ieee_tnsre_author.pdf)
- DOI: [10.1109/TNSRE.2020.3038322](https://doi.org/10.1109/TNSRE.2020.3038322)
- Metadata: Zhu, Martinez-Luna, Li, McDonald, Dai, Huang, Farrell, Clancy,
  *IEEE Transactions on Neural Systems and Rehabilitation Engineering*
  28(12):3040-3050, 2020. Volume, issue and pages were confirmed against both
  the author's copy and the Crossref record.
- Verified claim: in a forearm target-tracking study using a 16-channel bipolar
  electrode array on 12 able-bodied and 7 unilateral transradial limb-absent
  subjects, EMG-force model error averaged about 10 %MVC in the best case
  (able-bodied dominant limb, unilateral, **with** force feedback), 12-16 %MVC
  when a bilateral tracking source supplied the model output, and 25-30 %MVC
  in the **no-feedback** condition. The paper states that the no-feedback error
  was nearly half the tested force range of +/- 30 %MVC and concludes that the
  no-feedback model output "was not acceptable."
- Claim boundary: this is an offline system-identification result for EMG-force
  regression models under target tracking. It is not a robot control-loop
  evaluation, its error is a force/moment model error rather than an angular
  positioning error, and its 16-channel gelled laboratory array is not this
  project's three-channel band. It establishes that open-loop, no-feedback
  proportional amplitude decoding is poor; it does **not** establish that
  velocity control with visual feedback fixes it, and it says nothing about
  this project's activation threshold, calibration, or event gate.
- Use: Objective 3.5 proportional view-control design decision -- the reason
  the `activation` field drives a continuous velocity command the wearer closes
  visually, rather than a one-shot amplitude-to-angle step. This project's own
  no-feedback repeatability, computed from the three per-gesture trials in
  `datasets/emg_calibration/calibration_20260815_155306.json`, is consistent
  with the paper's no-feedback figure and is not yet written up in a log.

### `scheme2014motionnormalized`

- Type: research paper
- Primary source: [PubMed record](https://pubmed.ncbi.nlm.nih.gov/23475378/)
  and [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/6473893)
- DOI: [10.1109/TNSRE.2013.2247421](https://doi.org/10.1109/TNSRE.2013.2247421)
- Metadata: Scheme, Lock, Hargrove, Hill, Kuruganti, Englehart, *IEEE
  Transactions on Neural Systems and Rehabilitation Engineering*, 2014
  Jan;22(1):149-57.
- Verified claim: two proportional-control algorithms for pattern
  recognition-based myoelectric control that automatically configure
  **motion-specific gains** and normalize the control space to the user's
  usable dynamic range, with **class-specific normalization parameters
  computed from the data collected during classifier training**, requiring no
  additional user action. Reported improvements over the incumbent method of
  21% for amputee and 40% for able-bodied subjects.
- Read: abstract and bibliographic record only. The full text has not been
  read, so no experimental condition, metric definition, or number from it may
  enter project prose beyond what is written above.
- Claim boundary: the improvement figures come from the abstract and their
  experimental conditions have not been checked here.
- Use: Objective 3.5 -- **this is the prior art for what this project measures
  per direction.** Measuring a full-deflection reference for each steering
  gesture and normalizing that gesture's activation against its own reference
  is class-specific normalization, published in 2014. This project derives its
  references from a separate per-donning calibration capture rather than from
  the classifier training data, which is a variant of the procedure and not a
  new idea. Recorded so no part of the reference-level work is described as
  novel.

### `hwang2017robustness`

- Type: research paper
- Primary source: [PLOS ONE article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0186318)
- DOI: [10.1371/journal.pone.0186318](https://doi.org/10.1371/journal.pone.0186318)
- Metadata: Hwang, Hahne, Muller, *PLOS ONE*, 2017. Confirmed against Crossref.
- Verified claim: in real-time regression-based (proportional) myoelectric
  control with 16 forearm channels -- two circles, 35 mm spacing, at about a
  third of the elbow-to-wrist distance -- donning/doffing caused a significant
  performance decrease across all metrics, while arm position change caused no
  significant loss online. The paper concludes that arm position change is "of
  lesser critical concern in practical control situations, but mechanical or
  algorithmic solutions are needed to resolve the negative impact of
  donning/doffing".
- Claim boundary: 16 channels against this project's three, and able-bodied
  subjects on a cursor task rather than a robot. It establishes which of the
  two confounds dominates for proportional control; it does not say what a
  three-channel band can achieve, and its arm-position finding concerns limb
  posture during use, not the starting posture a gesture is performed from.
- Use: Objective 3.5 -- the reason this project's cross-donning degradation is
  treated as the expected hard problem rather than a collection error, and the
  reason re-donning is the variable worth controlling.

### `olsson2021mrl`

- Type: research paper
- Primary source: [PubMed Central full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7885418/)
- DOI: [10.1186/s12984-021-00832-4](https://doi.org/10.1186/s12984-021-00832-4)
- Metadata: Olsson, Malesevic, Bjorkman, Antfolk, *Journal of NeuroEngineering
  and Rehabilitation*, 2021. Confirmed against Crossref.
- Verified claim: eight circularly arranged dry surface electrodes on the
  forearm, at about a third of the elbow-to-wrist distance, drove simultaneous
  and proportional control learned from **categorical movement labels only**,
  with no continuous ground truth. Movements were encoded ternary (rest plus
  eight compound wrist/digit movements as targets such as [-1, 0] or [1, 1])
  and a shared-encoder, multi-branch network regressed sEMG envelopes onto
  continuous kinematics. Evaluated online with 20 able-bodied subjects on a
  Fitts-law cursor task, on day 1 and again on day 7 without recalibration,
  with no significant deterioration; re-donning was guided by photographs.
  Reported inference cost is 5.5 MFLOPS and a 59.5 kB memory footprint.
- Claim boundary: eight channels against this project's three, and the reported
  week-long stability is attributed partly to the electrodes' large pickup
  area, so it does not transfer to a three-channel band unexamined. The
  59.5 kB footprint does not fit an STM32F103C8T6 at all: 20 kB RAM total and
  about 20 kB of flash left after the current firmware. The paper does not
  claim its architecture is necessary for the categorical-label result.
- Use: Objective 3.5 -- the feasibility precedent that kept proportional
  control from categorical labels on the table when it was in question. No
  method, architecture, or result from this paper is used in the
  implementation: direction comes from the classifier class and magnitude from
  the envelope, which is not their regression network. Their photograph-guided
  re-donning was adopted in intent -- photographs are taken -- but no protocol
  uses them yet. Their day-1/day-7 result without recalibration is the
  contrast this project cannot claim: every session here is recalibrated, and
  contact was measured decaying within hours.
- Use: Objective 3.5 -- the source for training a proportional output from the
  discrete gesture labels this project already records, which would remove the
  need for a measured reference level and the downlink to carry it. The
  network itself is not portable here; whether a linear regressor onto the same
  ternary encoding reproduces the useful part is this project's own question,
  not something this paper answers.

## Candidate queue

These sources are deliberately not in `references.bib` and must not yet be
used as factual support.

| Candidate | Why it may matter | Verification still required |
| --- | --- | --- |
| [Rolling Shutter Camera Synchronization with Sub-millisecond Accuracy](https://arxiv.org/abs/1902.11084) | May support the distinction between timestamp pairing and exposure synchronization for ordinary USB cameras. | Verify final venue and metadata; read the method/limitations against this project's stop-and-look use. |
| [Multi-View Picking: Next-best-view Reaching for Improved Grasping in Clutter](https://arxiv.org/abs/1809.08564) | May establish active viewpoint selection as prior work for Objective 5 refinement/search. | Verify final publication record and identify the exact baseline claim needed. |
| [Olsson et al. 2021 companion question: does a *linear* regressor onto a ternary encoding work?](https://doi.org/10.1186/s12984-021-00832-4) | The categorical-labels-to-proportional-output result is the transferable part; the network is not. A ridge regression onto the same encoding would cost less than the current 4-class LDA. | Not a literature question -- this needs measuring on this project's own recordings before any claim. Listed here so it is not mistaken for something the paper established. |
| [Smith, Kuiken, Hargrove — Linear regression simultaneous myoelectric control](https://doi.org/10.1109/TBME.2015.2469741) | Regression rather than classification for simultaneous proportional control. | **Intramuscular** EMG, so channel count and signal quality are not comparable; read before assuming any of it transfers to a surface band. |
| [Shafieian & Nougarou — Two-stage regression structure, EMBC 2023](https://doi.org/10.1109/EMBC40787.2023.10340870) | Detects the DoF first, then selects a per-direction regression model — structurally close to "which direction, then how far". | Conference paper; read the full text for channel count and whether the two-stage split buys anything at three channels. |
| Hand-eye calibration primary source | Needed before formal claims about `base -> end_effector -> stereo_reference` calibration. | Select and read an appropriate primary method/comparison rather than citing an OpenCV API alone. |
| [ISO 13482 — Personal care robots](https://www.iso.org/standard/53820.html) | The **nearest-scoped** standard for an assistive device, and therefore the one to read first — not the industrial pair below. | Confirm current edition and scope. Read which hazard classes actually apply to a fixed-delivery-zone handoff before any design text leans on it. |
| [ISO/TS 15066 — Collaborative robots](https://www.iso.org/standard/62996.html) | Sometimes cited for power/force limiting in human-robot contact. | Scoped to **industrial** robots; confirm whether anything in it legitimately transfers to a service/assistive context, or whether citing it would overclaim. |
| [ISO 10218-2 — Industrial robot systems and integration](https://www.iso.org/standard/73934.html) | Sometimes cited for whole-system safety integration. | Same industrial-scope caveat. Verify the 2025 edition's status and whether it explicitly excludes service/personal-care robots. |

None of the three standards above may be used to imply that this project is
safety-certified or safety-rated. The stereo hand cue is documented as an
explicitly non-safety-rated signal, and that stays true regardless of which
standard is eventually read.

## Maintenance checklist

- Add a source here before adding it to project prose.
- Record only the smallest claim directly supported by the source.
- Record what the source does **not** establish for this project.
- Prefer the archival paper over a blog, search result, or secondary citation.
- Use official documentation for APIs and contracts, not research novelty.
- Keep cite keys stable and unique.
- Re-check online-first versus issue year before entering journal metadata.
- Do not copy experimental numbers into project claims without reading the
  corresponding experiment and conditions.
