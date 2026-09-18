package com.bbfc.notification.core.port;

import com.bbfc.notification.core.domain.EventId;

public interface NotificationChannel {

    /** Sends an alert carrying an acknowledge button. Returns the id of the message sent. */
    long sendAlert(EventId eventId, String message);

    /** Sends the triage question carrying genuine / false-alarm buttons. Returns the message id. */
    long sendTriageQuestion(EventId eventId, String message);

    /** Best effort: replaces a message's text and clears its keyboard. Never throws. */
    void closeMessage(long messageId, String text);

    /**
     * Best effort: clears the spinner on a tapped button, ideally within ~15s of the tap.
     * Never throws — a spinning button must not cost us the acknowledgement it was recording.
     */
    void answerCallback(String callbackQueryId, String text);
}
