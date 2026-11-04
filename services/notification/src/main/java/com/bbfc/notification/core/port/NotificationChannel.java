package com.bbfc.notification.core.port;

import com.bbfc.notification.core.domain.EventId;

import java.util.Optional;

public interface NotificationChannel {

    /** Sends an alert carrying an acknowledge button. Returns the id of the message sent. */
    long sendAlert(EventId eventId, String message);

    /**
     * Best effort: sends the alert as the clip, captioned and carrying an acknowledge button.
     * Never throws. Empty means Telegram refused it.
     */
    Optional<ClipDelivery> sendAlertWithClip(EventId eventId, String caption, ClipContent content);

    /**
     * Sends the triage question carrying genuine / false-alarm buttons, as a reply to the alert
     * message when there is one. Returns the message id.
     */
    long sendTriageQuestion(EventId eventId, String message, Long replyToMessageId);

    /** Sends a keyboardless escalation reply to the alert message when there is one. Returns the message id. */
    long sendEscalation(String message, Long replyToMessageId);

    /**
     * Best effort: replaces a message's text, or its caption when it carries a clip, and clears
     * its keyboard. Never throws.
     */
    void closeMessage(long messageId, String text, boolean carriesClip);

    /**
     * Best effort: clears the spinner on a tapped button, ideally within ~15s of the tap.
     * Never throws — a spinning button must not cost us the acknowledgement it was recording.
     */
    void answerCallback(String callbackQueryId, String text);
}
