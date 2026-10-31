# Camera Heads are promoted without human review

A Refit runs every week, and a Candidate Head that passes the Promotion Gate becomes the Camera Head every deployment runs, through a pull request that merges itself once CI is green. Nobody approves it. A weekly human sign-off was the alternative, rejected so the loop runs unattended. The Promotion Gate is therefore the only safeguard, and it is built to be strict: a Candidate Head that loses even one Genuine fall on the held-out picam session or the Eval Alerts is rejected, however many False Alarms it removes, and it has to remove at least two, which is more than the Seed Floor.

## Consequences

- A head that passes the gate but behaves worse in the room runs until someone notices. Rollback is reverting the Promotion's pull request, which ships the previous head through the same CI path.
- The record of a Promotion is the pull request and the Actions run, not the Telegram chat, which is for responders only.
- Only the Camera Head is promoted this way. A new Checkpoint or Pose Backbone still goes through a reviewed pull request.
