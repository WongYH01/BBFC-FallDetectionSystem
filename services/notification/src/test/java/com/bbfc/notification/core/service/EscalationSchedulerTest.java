package com.bbfc.notification.core.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.BDDMockito.given;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.AlertState;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EscalationPolicy;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.FixedEscalationPolicy;
import com.bbfc.notification.core.domain.RoomRef;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.MessageRenderer;
import com.bbfc.notification.core.port.NotificationChannel;
import com.bbfc.notification.testsupport.MutableClock;


@ExtendWith(MockitoExtension.class)
public class EscalationSchedulerTest {
    @Mock
    private AlertRepository alertRepository;

    @Mock
    private NotificationChannel notificationChannel;

    @Mock
    private MessageRenderer messageRenderer;

    private final EscalationPolicy policy = new FixedEscalationPolicy(Duration.ofSeconds(60), 3);
    private final MutableClock clock = new MutableClock(Instant.parse("2026-09-16T10:00:00Z"), ZoneOffset.UTC);

    private EscalationScheduler scheduler;
    private EventId eventId;
    private RoomRef room;
    private Confidence confidence;

    @BeforeEach
    void setUp() {
        scheduler = new EscalationScheduler(alertRepository, notificationChannel, messageRenderer, policy, clock);
        eventId = new EventId("FE-1");
        room = new RoomRef("room-12", "Block A - Room 12");
        confidence = new Confidence(0.9);
    }

    @Test
    void noDueAlertsDoesNothing() {
        given(alertRepository.findDueForEscalation(any(), anyInt())).willReturn(List.of());
        scheduler.escalateDueAlerts();
        verify(notificationChannel, never()).sendAlert(any());
        verify(alertRepository, never()).save(any());
    }

    @Test
    void dueAlertBelowCapEscalatesAndReschedules() {
        given(alertRepository.save(any())).willAnswer(invocation -> invocation.getArgument(0));
        Alert alert = Alert.dispatch(eventId, room, confidence);
        alert.scheduleNextEscalation(clock.instant());
        given(alertRepository.findDueForEscalation(any(), anyInt())).willReturn(List.of(alert));
        given(messageRenderer.render(any(), any())).willReturn("escalation message");

        scheduler.escalateDueAlerts();

        assertThat(alert.state()).isEqualTo(AlertState.ESCALATING);
        assertThat(alert.repeatCount()).isEqualTo(1);
        assertThat(alert.nextEscalationAt()).contains(clock.instant().plus(Duration.ofSeconds(60)));
        verify(notificationChannel).sendAlert("escalation message");
        verify(alertRepository).save(alert);
    }

    @Test
    void dueAlertAtCapExhaustsWithoutSending() {
        given(alertRepository.save(any())).willAnswer(invocation -> invocation.getArgument(0));
        Alert alert = Alert.dispatch(eventId, room, confidence);
        alert.escalate(clock.instant());
        alert.escalate(clock.instant());
        alert.escalate(clock.instant());
        alert.scheduleNextEscalation(clock.instant());
        given(alertRepository.findDueForEscalation(any(), anyInt())).willReturn(List.of(alert));

        scheduler.escalateDueAlerts();

        assertThat(alert.state()).isEqualTo(AlertState.EXHAUSTED);
        verify(notificationChannel, never()).sendAlert(any());
        verify(alertRepository).save(alert);
    }

    @Test
    void repeatsEscalateToCapThenExhausts() {
        given(alertRepository.save(any())).willAnswer(invocation -> invocation.getArgument(0));
        Alert alert = Alert.dispatch(eventId, room, confidence);
        alert.scheduleNextEscalation(clock.instant().plus(Duration.ofSeconds(60)));
        given(alertRepository.findDueForEscalation(any(), anyInt())).willReturn(List.of(alert));
        given(messageRenderer.render(any(), any())).willReturn("escalation message");

        clock.advance(Duration.ofSeconds(60));
        scheduler.escalateDueAlerts();
        assertThat(alert.repeatCount()).isEqualTo(1);
        assertThat(alert.state()).isEqualTo(AlertState.ESCALATING);

        clock.advance(Duration.ofSeconds(60));
        scheduler.escalateDueAlerts();
        assertThat(alert.repeatCount()).isEqualTo(2);

        clock.advance(Duration.ofSeconds(60));
        scheduler.escalateDueAlerts();
        assertThat(alert.repeatCount()).isEqualTo(3);

        verify(notificationChannel, times(3)).sendAlert(any());

        clock.advance(Duration.ofSeconds(60));
        scheduler.escalateDueAlerts();

        assertThat(alert.state()).isEqualTo(AlertState.EXHAUSTED);
        verify(notificationChannel, times(3)).sendAlert(any());
    }
}
