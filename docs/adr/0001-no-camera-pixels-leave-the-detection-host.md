# Camera pixels are never stored or sent off the detection host

The system watches people in a room where they live or are cared for, so it is built to be privacy-preserving: everything after the Pose Backbone runs on keypoints, and the only video that leaves the detection host is a Skeleton Clip, which is the pose drawn on a black background. The detection service does not keep footage. It deletes uploaded clips (`KEEP_UPLOADED_CLIPS=0`) and refuses the manual Record button, which is the one path that writes raw camera video (`ALLOW_MANUAL_RECORDING=0`). The notification service's storage bucket holds skeleton clips only.

## Consequences

- A responder cannot see the room itself in an Alert. Whether a fall is Genuine has to be judged from the skeleton alone, which is also all the classifier ever sees.
- Footage of a false alarm is lost unless a human recreates it; its keypoints are kept for training (ADR 0002). Commissioning clips still have to be staged and recorded on purpose with demo-cam, never taken from the service.
- Any feature that needs pixels downstream, such as scene context to tell a bed from the floor, cuts against this decision and needs a new ADR, even though it is the known fix for lying-down false alarms.
