package com.bbfc.notification.core.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
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
import com.bbfc.notification.core.domain.AlertState;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.Outcome;
import com.bbfc.notification.core.domain.RoomRef;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.MessageRenderer;
import com.bbfc.notification.core.port.NotificationChannel;

@ExtendWith(MockitoExtension.class)
class TriageAlertServiceTest {

    private static final Instant DETECTED = Instant.parse("2026-09-16T10:00:00Z");
    private static final Instant TAPPED = Instant.parse("2026-09-16T10:05:00Z");
    private static final String CALLBACK_ID = "cbq-2";

    @Mock
    private AlertRepository alertRepository;

    @Mock
    private NotificationChannel notificationChannel;

    @Mock
    private MessageRenderer messageRenderer;

    private TriageAlertService service;
    private EventId eventId;

    @BeforeEach
    void setUp() {
        service = new TriageAlertService(alertRepository, notificationChannel, messageRenderer);
        eventId = new EventId("FE-1");
    }

    private Alert acknowledgedAlert() {
        Alert alert = Alert.dispatch(
                eventId, new RoomRef("room-12", "Block A - Room 12"), new Confidence(0.92), DETECTED);
        alert.acknowledge("@nurse_lim", DETECTED);
        alert.recordFollowUpMessage(504L);
        return alert;
    }

    @Test
    void recordsTheOutcomeAndClosesTheFollowUpMessage() {
        Alert alert = acknowledgedAlert();
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.of(alert));
        given(messageRenderer.renderTriaged(alert)).willReturn("🚫 Recorded: False alarm");

        service.triage(CALLBACK_ID, eventId, Outcome.FALSE_ALARM, TAPPED);

        assertThat(alert.state()).isEqualTo(AlertState.TRIAGED);
        assertThat(alert.outcome()).contains(Outcome.FALSE_ALARM);
        assertThat(alert.outcomeAt()).contains(TAPPED);
        verify(alertRepository).save(alert);
        verify(notificationChannel).answerCallback(CALLBACK_ID, "Recorded");
        verify(notificationChannel).closeMessage(504L, "🚫 Recorded: False alarm");
    }

    @Test
    void recordsAGenuineFall() {
        Alert alert = acknowledgedAlert();
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.of(alert));
        given(messageRenderer.renderTriaged(alert)).willReturn("👍 Recorded: Genuine fall");

        service.triage(CALLBACK_ID, eventId, Outcome.GENUINE, TAPPED);

        assertThat(alert.outcome()).contains(Outcome.GENUINE);
    }

    @Test
    void secondTapIsRejectedWithTheRecordedOutcome() {
        Alert alert = acknowledgedAlert();
        alert.triage(Outcome.GENUINE, DETECTED);
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.of(alert));

        service.triage(CALLBACK_ID, eventId, Outcome.FALSE_ALARM, TAPPED);

        assertThat(alert.outcome()).contains(Outcome.GENUINE);
        verify(notificationChannel).answerCallback(CALLBACK_ID, "Already recorded as a genuine fall");
        verify(alertRepository, never()).save(any());
        verify(notificationChannel, never()).closeMessage(anyLong(), any());
    }

    @Test
    void triageBeforeAcknowledgementIsRejected() {
        Alert alert = Alert.dispatch(
                eventId, new RoomRef("room-12", "Block A - Room 12"), new Confidence(0.92), DETECTED);
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.of(alert));

        service.triage(CALLBACK_ID, eventId, Outcome.GENUINE, TAPPED);

        assertThat(alert.state()).isEqualTo(AlertState.DISPATCHED);
        verify(notificationChannel).answerCallback(CALLBACK_ID, "That alert has not been acknowledged yet.");
        verify(alertRepository, never()).save(any());
    }

    @Test
    void unknownEventIdIsAnsweredNotThrown() {
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.empty());

        service.triage(CALLBACK_ID, eventId, Outcome.GENUINE, TAPPED);

        verify(notificationChannel).answerCallback(CALLBACK_ID, "That alert is no longer tracked.");
        verify(alertRepository, never()).save(any());
    }
}
