package com.bbfc.notification.core.domain;

import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class AlertTest {

    private Alert alert;

    @BeforeEach 
    void setUp(){
        alert = Alert.dispatch(
            new EventId("EE-01"), 
            new RoomRef("room-1", "Block A, Room 1"), 
            new Confidence(0.9)
        );
    }

    @Nested
    @DisplayName("legal transitions")
    class LegalTransitions{
        @Test 
        void acknowledgeFromDispatchSucceeds(){
            alert.acknowledge("nurse-1", Instant.now());
            assertThat(alert.state()).isEqualTo(AlertState.ACKNOWLEDGED);
        }

        @Test 
        void escalateFromDispatchSucceeds(){
            alert.escalate(Instant.now());
            assertThat(alert.state()).isEqualTo(AlertState.ESCALATING);
        }

        @Test
        void triageAfterAcknowledgeSucceeds() {
            alert.acknowledge("nurse-1", Instant.now());
            alert.triage(Outcome.GENUINE, Instant.now());
            assertThat(alert.state()).isEqualTo(AlertState.TRIAGED);
        }

        @Test
        void acknowledgeWhileEscalatingSucceeds() {
            alert.escalate(Instant.now());
            alert.acknowledge("nurse-1", Instant.now());
            assertThat(alert.state()).isEqualTo(AlertState.ACKNOWLEDGED);
        }

        @Test 
        void escalateStaysEscalatingSucceeds(){
            alert.escalate(Instant.now());
            alert.escalate(Instant.now());
            assertThat(alert.state()).isEqualTo(AlertState.ESCALATING);
        }

        @Test 
        void exhaustAfterEscalatingSucceeds(){
            alert.escalate(Instant.now());
            alert.exhaust(Instant.now());
            assertThat(alert.state()).isEqualTo(AlertState.EXHAUSTED);
        }


    }

    @Nested 
    @DisplayName("illegal transitions")
    class IllegalTransitions{
        @Test 
        void triageBeforeAcknowledgeError(){
            assertThatThrownBy(()->alert.triage(Outcome.GENUINE, Instant.now())).isInstanceOf(IllegalAlertTransitionException.class);
        }

        @Test 
        void acknowledgeTwiceError(){
            alert.acknowledge("nurse-1", Instant.now());
            assertThatThrownBy(()->alert.acknowledge("nurse-2", Instant.now())).isInstanceOf(IllegalAlertTransitionException.class);
        }

        @ParameterizedTest
        @EnumSource(value = AlertState.class, names = {"EXHAUSTED", "TRIAGED"})
        void terminalStatesRejectEveryTransition(AlertState terminal) {
            assertThat(terminal.canTransitionTo(AlertState.ACKNOWLEDGED)).isFalse();
            assertThat(terminal.canTransitionTo(AlertState.ESCALATING)).isFalse();
            assertThat(terminal.canTransitionTo(AlertState.TRIAGED)).isFalse();
            assertThat(terminal.canTransitionTo(AlertState.EXHAUSTED)).isFalse();
        }

    }

    @Nested
    @DisplayName("escalation scheduling")
    class EscalationScheduling {

        @Test
        void newAlertHasNoEscalationScheduledByDefault() {
            assertThat(alert.nextEscalationAt()).isEmpty();
            assertThat(alert.repeatCount()).isZero();
        }

        @Test
        void scheduleNextEscalationSetsTheDueTime() {
            Instant due = Instant.parse("2026-09-16T10:01:00Z");
            alert.scheduleNextEscalation(due);
            assertThat(alert.nextEscalationAt()).contains(due);
        }

        @Test
        void escalateIncrementsRepeatCount() {
            alert.escalate(Instant.now());
            alert.escalate(Instant.now());
            assertThat(alert.repeatCount()).isEqualTo(2);
        }

        @Test
        void exhaustClearsNextEscalation() {
            alert.scheduleNextEscalation(Instant.now());
            alert.escalate(Instant.now());
            alert.exhaust(Instant.now());
            assertThat(alert.nextEscalationAt()).isEmpty();
        }

        @Test
        void acknowledgeClearsNextEscalation() {
            alert.scheduleNextEscalation(Instant.now());
            alert.acknowledge("nurse-1", Instant.now());
            assertThat(alert.nextEscalationAt()).isEmpty();
        }
    }

}
