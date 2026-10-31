# BBFC Fall Detection

Watches one room through one camera, decides from body pose whether someone has fallen, and gets a human to respond. Starts from a replication of a published pose-transformer paper; the decision layer has since been rebuilt for a real deployment.

## Seeing the person

**Pose Backbone**:
The frozen pose model that turns each video frame into 17 body keypoints. It is never trained here.
_Avoid_: detector, YOLO (when you mean the role rather than the product)

**Keypoint Confidence**:
The pose backbone's certainty that a keypoint is where it says. Not the same thing as an Alert's Confidence.
_Avoid_: confidence (unqualified)

**Dropout**:
Frames, or the share of a window's frames, in which the pose backbone found nobody.
_Avoid_: missed detection, gap

**Subject**:
A person the pipeline follows and decides on, with their own keypoint buffer and alarm. Each tracked person is a subject, and the room's alarm is up while any subject's is. A subject is not a tracker id: it can outlive one (see Reconnect). Under `FALL_SUBJECTS=largest` there is one subject, the largest person box.
_Avoid_: patient, user, person (when you mean the one being tracked), track (when you mean the subject rather than the tracker's id)

**Phantom Subject**:
A Subject that is not a person: furniture or clutter the pose backbone took for one. A false alarm it raises is not an ADL.
_Avoid_: ghost, false detection, furniture (when you mean the subject)

**Reconnect**:
A new tracker id continuing the stream of a subject lost moments ago nearby, rather than starting a new subject. The tracker re-identifies a faller in about 40% of falls, mid-descent.
_Avoid_: re-identification, stitching, merge

## Deciding

**Window**:
A fixed-length run of a subject's keypoints, about 4 seconds, that the classifier scores as one unit. A new one is scored every second.
_Avoid_: clip, sequence, segment

**Checkpoint**:
One trained classifier that turns a window into P(fall).
_Avoid_: model (when several are in play)

**Camera Head**:
A small classifier fitted on top of a checkpoint's internal features, using one camera's own footage as that camera is currently mounted. Re-aiming the camera changes its footage, so the head has to be refitted.
_Avoid_: room head, calibration head, probe, head (unqualified)

**Commissioning Recording**:
The staged clips, falls and everyday activities both, recorded on one camera, so its camera head can be fitted.
_Avoid_: calibration set, training clips

**Fall Decision**:
The full rule a camera runs: the checkpoint(s) and their camera heads, plus the rolling buffer, threshold, latch and descent gate that turn window scores into an alarm. Live, replay and offline scoring all use the same one.
_Avoid_: model, detector, pipeline

**Rolling Buffer**:
The last few window scores, averaged, that the alarm threshold is compared against.
_Avoid_: smoothing, history

**Latch**:
The alarm staying raised once it is raised, until the rolling score has stayed low for a set number of windows.
_Avoid_: hysteresis (as a noun for the state), persistence

**Descent Gate**:
The requirement that a window contain a real downward hip movement before it may raise the alarm. It does not affect clearing.
_Avoid_: motion gate, velocity gate

**Detection Gate**:
An evaluation-time rule: a window the subject is mostly absent from reports "no subject" rather than a verdict. It is not part of the Fall Decision a camera runs.
_Avoid_: dropout gate, max_dropout (outside the code)

**Alarm**:
The detection side's latched state, meaning "this room's fall decision currently says Fall". One alarm produces one Alert.
_Avoid_: alert (on the detection side), trigger

## Activities and data

**ADL**:
An everyday activity that is not a fall (walking, sitting down, lying down, tying a shoelace). Most false alarms are an ADL the system called a fall; the rest come from a Phantom Subject.
_Avoid_: negative, non-fall, normal activity

**Controlled Descent**:
An ADL that takes the body from upright to low or horizontal on purpose, such as lying down or sitting on the floor. These are the hardest ADLs to tell from a fall, and the scarcest in training data.
_Avoid_: transition, near-fall

**Hard Negative**:
An ADL clip added to training specifically because it looks like a fall.
_Avoid_: hard example, adversarial negative

**picam**:
The project's own recordings from the deployment room. **picam-1** and **picam-2** are two camera positions in the *same* room, not two rooms.
_Avoid_: deployment set, our dataset, second room (for picam-2)

**LOSO**:
Leave-one-subject-out scoring: fit on every recorded subject but one and score the one left out. The honest number for anything fitted on picam.
_Avoid_: cross-validation (unqualified)

**Seed Floor**:
The spread in a metric from changing only the random seed. A difference smaller than this has not been measured.
_Avoid_: noise (unqualified), variance

## Alerting

**Alert**:
The notification service's record of one fall incident, from the moment it is dispatched until a human has triaged it.
_Avoid_: event, notification, incident, alarm (on the notification side)

**Event ID**:
The identifier the detection side gives one alarm. The notification service creates at most one Alert per event ID, which makes resending safe.
_Avoid_: alert ID, incident ID

**Confidence**:
The P(fall) attached to an Alert, read by humans as low, medium or high. Not the same thing as a keypoint confidence.
_Avoid_: score, probability (in alerting)

**Dispatch**:
Sending an Alert's first message to the responders' Telegram chat.
_Avoid_: send, notify

**Escalation**:
Re-sending an unacknowledged Alert on a fixed interval, up to a set number of repeats.
_Avoid_: reminder, retry, nag

**Exhausted**:
An Alert that ran out of escalation repeats without being acknowledged. It can still be acknowledged late.
_Avoid_: expired, failed, timed out

**Acknowledge**:
A responder's claim that they are handling an Alert. It stops escalation.
_Avoid_: accept, confirm, respond

**Triage**:
A responder's verdict on an acknowledged Alert: Genuine or False Alarm. It closes the Alert.
_Avoid_: resolve, close, label

**Outcome**:
The result of triage, either **Genuine** or **False Alarm**. It is about the subject whose alarm raised the Alert, not any other subject seen in the same Skeleton Clip.
_Avoid_: verdict, resolution

**Skeleton Clip**:
The video sent with an Alert, showing the pose drawn on a black background. It contains no camera pixels.
_Avoid_: fall clip, recording, footage, video

**Keypoint File**:
The keypoints of everyone tracked across the same frames as an Alert's Skeleton Clip, kept so the Alert's Outcome can be trained on.
_Avoid_: pose file, skeleton data, keypoints (unqualified)

## Refitting

**Refit**:
Fitting a new Camera Head on the Commissioning Recording plus the camera's triaged Alerts, for the deployed checkpoint and pose backbone. It happens once a week, and is skipped when too few new Alerts have been triaged. The checkpoint is never retrained.
_Avoid_: retrain, model update

**Candidate Head**:
The Camera Head a Refit produces, before the Promotion Gate has judged it.
_Avoid_: new model, new head

**Eval Alert**:
A triaged Alert set aside, for good, for judging Candidate Heads. No Refit ever trains on it.
_Avoid_: test alert, validation set

**Promotion Gate**:
The rule a Candidate Head must pass to replace the deployed one: scored with the full Fall Decision on the held-out picam session and the Eval Alerts, it loses no Genuine fall and raises at least two fewer False Alarms.
_Avoid_: accuracy threshold, quality gate

**Promotion**:
A Candidate Head that passed the Promotion Gate becoming the Camera Head every deployment runs.
_Avoid_: deploy, release (for the head)
