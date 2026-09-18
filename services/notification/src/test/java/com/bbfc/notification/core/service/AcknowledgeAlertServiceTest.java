package com.bbfc.notification.core.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
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
import org.springframework.context.ApplicationEventPublisher;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.AlertState;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.RoomRef;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.MessageRenderer;
import com.bbfc.notification.core.port.NotificationChannel;

@ExtendWith(MockitoExtension.class)
class AcknowledgeAlertServiceTest {

    private static final Instant DETECTED = Instant.parse("2026-09-16T10:00:00Z");
    private static final Instant TAPPED = Instant.parse("2026-09-16T10:03:00Z");
    private static final String CALLBACK_ID = "cbq-1";

    @Mock
    private AlertRepository alertRepository;

    @Mock
    private NotificationChannel notificationChannel;

    @Mock
    private MessageRenderer messageRenderer;

    @Mock
    private ApplicationEventPublisher eventPublisher;

    private AcknowledgeAlertService service;
    private EventId eventId;

    @BeforeEach
    void setUp() {
        service = new AcknowledgeAlertService(
                alertRepository, notificationChannel, messageRenderer, eventPublisher);
        eventId = new EventId("FE-1");
    }

    private Alert dispatchedAlert() {
        return Alert.dispatch(
                eventId, new RoomRef("room-12", "Block A - Room 12"), new Confidence(0.92), DETECTED);
    }

    @Test
    void acknowledgesClosesEveryMessageAndPublishes() {
        Alert alert = dispatchedAlert();
        alert.recordDispatchMessage(501L);
        alert.recordEscalationMessage(502L);
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.of(alert));
        given(messageRenderer.renderAcknowledged(alert)).willReturn("HANDLED");

        service.acknowledge(CALLBACK_ID, eventId, "@nurse_lim", TAPPED);

        assertThat(alert.state()).isEqualTo(AlertState.ACKNOWLEDGED);
        assertThat(alert.acknowledgedBy()).contains("@nurse_lim");
        assertThat(alert.nextEscalationAt()).isEmpty();
        verify(alertRepository).save(alert);
        verify(notificationChannel).answerCallback(CALLBACK_ID, "Acknowledged");
        verify(notificationChannel).closeMessage(501L, "HANDLED");
        verify(notificationChannel).closeMessage(502L, "HANDLED");
        verify(eventPublisher).publishEvent(new AlertAcknowledgedEvent(eventId, "@nurse_lim", TAPPED));
    }

    @Test
    void lateTapOnAnExhaustedAlertIsStillRecorded() {
        Alert alert = dispatchedAlert();
        alert.escalate(DETECTED);
        alert.exhaust(DETECTED);
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.of(alert));
        given(messageRenderer.renderAcknowledged(alert)).willReturn("HANDLED");

        service.acknowledge(CALLBACK_ID, eventId, "@nurse_lim", TAPPED);

        assertThat(alert.state()).isEqualTo(AlertState.ACKNOWLEDGED);
        verify(eventPublisher).publishEvent(any(AlertAcknowledgedEvent.class));
    }

    @Test
    void secondTapIsRejectedWithWhoAlreadyHandledIt() {
        Alert alert = dispatchedAlert();
        alert.acknowledge("@nurse_lim", DETECTED);
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.of(alert));

        service.acknowledge(CALLBACK_ID, eventId, "@nurse_tan", TAPPED);

        assertThat(alert.acknowledgedBy()).contains("@nurse_lim");
        verify(notificationChannel).answerCallback(CALLBACK_ID, "Already handled by @nurse_lim");
        verify(alertRepository, never()).save(any());
        verify(notificationChannel, never()).closeMessage(anyLong(), any());
        verify(eventPublisher, never()).publishEvent(any(AlertAcknowledgedEvent.class));
    }

    @Test
    void unknownEventIdIsAnsweredNotThrown() {
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.empty());

        service.acknowledge(CALLBACK_ID, eventId, "@nurse_lim", TAPPED);

        verify(notificationChannel).answerCallback(CALLBACK_ID, "That alert is no longer tracked.");
        verify(alertRepository, never()).save(any());
        verify(eventPublisher, never()).publishEvent(any(AlertAcknowledgedEvent.class));
    }

    @Test
    void answersBeforeEditingSoTheSpinnerClearsFirst() {
        Alert alert = dispatchedAlert();
        alert.recordDispatchMessage(501L);
        given(alertRepository.findByEventIdForUpdate(eventId)).willReturn(Optional.of(alert));
        given(messageRenderer.renderAcknowledged(alert)).willReturn("HANDLED");

        service.acknowledge(CALLBACK_ID, eventId, "@nurse_lim", TAPPED);

        var inOrder = org.mockito.Mockito.inOrder(notificationChannel);
        inOrder.verify(notificationChannel).answerCallback(eq(CALLBACK_ID), any());
        inOrder.verify(notificationChannel).closeMessage(anyLong(), any());
    }
}
