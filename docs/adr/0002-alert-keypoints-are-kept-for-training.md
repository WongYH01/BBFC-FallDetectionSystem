# Every Alert's keypoints are kept for training

Each Alert carries a Keypoint File, the keypoints of everyone tracked over its Skeleton Clip, and the notification service keeps it, filed under the Alert's Outcome once it is triaged. A camera's false alarms are the Controlled Descents and Phantom Subjects of its own room as it is mounted, the examples training is shortest of, which ADR 0001 otherwise left to be staged and recorded by hand. Only keypoints leave the detection host, so ADR 0001 still holds.

## Consequences

- The Outcome is about the subject that raised the Alert (`raised_by` in the file), not everyone in it. The other subjects in a Genuine Alert are unlabelled: a carer kneeling beside the resident who fell is a Controlled Descent, and labelling it a fall would teach the model the false alarm it already makes. In a False Alarm nobody fell, so every subject in the file is an ADL or a Phantom Subject.
- Everyone tracked is kept, not only the subjects whose alarm went up, because the decision on one subject depends on the others (Reconnect, and a still subject that appears while someone is missing). Keeping them lets a new checkpoint or camera head be replayed over past Alerts exactly. The cost is that visitors' and carers' poses are stored off the detection host. Keypoints are far less identifying than video, but they are not anonymous.
- A Keypoint File whose Alert is never triaged stays unfiled and is not training data.
- The weekly Refit (ADR 0003) runs on GitHub-hosted runners in a public repository, so Keypoint Files are processed there too. They are downloaded, used and discarded: never uploaded as workflow artifacts or cached, and nothing in a log, run summary or pull request identifies an Alert. Only counts and a hash of the training set are published.
