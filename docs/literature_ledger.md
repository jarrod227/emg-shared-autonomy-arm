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

## Candidate queue

These sources are deliberately not in `references.bib` and must not yet be
used as factual support.

| Candidate | Why it may matter | Verification still required |
| --- | --- | --- |
| [Rolling Shutter Camera Synchronization with Sub-millisecond Accuracy](https://arxiv.org/abs/1902.11084) | May support the distinction between timestamp pairing and exposure synchronization for ordinary USB cameras. | Verify final venue and metadata; read the method/limitations against this project's stop-and-look use. |
| [Multi-View Picking: Next-best-view Reaching for Improved Grasping in Clutter](https://arxiv.org/abs/1809.08564) | May establish active viewpoint selection as prior work for Objective 5 refinement/search. | Verify final publication record and identify the exact baseline claim needed. |
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
