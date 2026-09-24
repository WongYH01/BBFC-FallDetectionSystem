package com.bbfc.notification.core.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.BDDMockito.given;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

import java.time.Instant;
import java.util.Optional;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.RoomRef;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.MessageRenderer;
import com.bbfc.notification.core.port.NotificationChannel;

@ExtendWith(MockitoExtension.class)
class FollowUpQuestionListenerTest {

    private static final Instant DETECTED = Instant.parse("2026-09-16T10:00:00Z");
    private static final Instant TAPPED = Instant.parse("2026-09-16T10:03:00Z");

    @Mock
    private AlertRepository alertRepository;

    @Mock
    private NotificationChannel notificationChannel;

    @Mock
    private MessageRenderer messageRenderer;

    private FollowUpQuestionListener listener;
    private EventId eventId;
    private AlertAcknowledgedEvent event;

    @BeforeEach
    void setUp() {
        listener = new FollowUpQuestionListener(alertRepository, notificationChannel, messageRenderer);
        eventId = new EventId("FE-1");
        event = new AlertAcknowledgedEvent(eventId, "@nurse_lim", TAPPED);
    }

    private Alert acknowledgedAlert() {
        Alert alert = Alert.dispatch(
                eventId, new RoomRef("room-12", "Block A - Room 12"), new Confidence(0.92), DETECTED);
        alert.acknowledge("@nurse_lim", TAPPED);
        return alert;
    }

    @Test
    void sendsTheQuestionAndRecordsItsMessageId() {
        Alert alert = acknowledgedAlert();
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.of(alert));
        given(messageRenderer.renderTriageQuestion(alert)).willReturn("Was this a genuine fall?");
        given(notificationChannel.sendTriageQuestion(eventId, "Was this a genuine fall?")).willReturn(504L);

        listener.onAlertAcknowledged(event);

        assertThat(alert.followUpMessageId()).contains(504L);
        verify(alertRepository).save(alert);
    }

    @Test
    void doesNotSendTwiceWhenTheColumnIsAlreadySet() {
        Alert alert = acknowledgedAlert();
        alert.recordFollowUpMessage(504L);
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.of(alert));

        listener.onAlertAcknowledged(event);

        verify(notificationChannel, never()).sendTriageQuestion(any(), any());
        verify(alertRepository, never()).save(any());
    }

    @Test
    void unknownEventIdIsIgnored() {
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.empty());

        listener.onAlertAcknowledged(event);

        verify(notificationChannel, never()).sendTriageQuestion(any(), any());
    }

    // Under MANDATORY an escaping exception would roll back the acknowledgement that published
    // this event, resuming escalation on an alert someone already handled.
    @Test
    void aFailedSendNeverEscapesAndSoCannotUndoTheAcknowledgement() {
        Alert alert = acknowledgedAlert();
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.of(alert));
        given(messageRenderer.renderTriageQuestion(alert)).willReturn("Was this a genuine fall?");
        given(notificationChannel.sendTriageQuestion(any(), any()))
                .willThrow(new RuntimeException("telegram down"));

        assertThatCode(() -> listener.onAlertAcknowledged(event)).doesNotThrowAnyException();

        assertThat(alert.followUpMessageId()).isEmpty();
    }
}
