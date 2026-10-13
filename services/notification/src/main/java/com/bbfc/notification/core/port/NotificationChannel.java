package com.bbfc.notification.core.port;

import com.bbfc.notification.core.domain.EventId;

import java.util.Optional;

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

    /**
     * Best effort: posts the clip as a reply to the alert message. Never throws — the text alert
     * has already been delivered and a clip failure must not undo it. Empty means Telegram
     * refused it.
     */
    Optional<ClipDelivery> sendClipReply(long replyToMessageId, ClipContent content);
}
