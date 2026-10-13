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
            new Confidence(0.9),
            Instant.parse("2026-09-16T10:00:00Z")
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
        @EnumSource(AlertState.class)
        void triagedRejectsEveryTransition(AlertState target) {
            assertThat(AlertState.TRIAGED.canTransitionTo(target)).isFalse();
        }

        @ParameterizedTest
        @EnumSource(value = AlertState.class, names = {"ESCALATING", "TRIAGED", "EXHAUSTED"})
        void exhaustedRejectsEverythingExceptALateAcknowledgement(AlertState target) {
            assertThat(AlertState.EXHAUSTED.canTransitionTo(target)).isFalse();
        }

        @Test
        void exhaustedAcceptsALateAcknowledgement() {
            alert.escalate(Instant.now());
            alert.exhaust(Instant.now());

            alert.acknowledge("nurse-1", Instant.now());

            assertThat(alert.state()).isEqualTo(AlertState.ACKNOWLEDGED);
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

    @Nested
    @DisplayName("message tracking")
    class MessageTracking {

        @Test
        void newAlertHasNoMessagesRecorded() {
            assertThat(alert.dispatchMessageId()).isEmpty();
            assertThat(alert.escalationMessageIds()).isEmpty();
            assertThat(alert.followUpMessageId()).isEmpty();
            assertThat(alert.messageIds()).isEmpty();
        }

        @Test
        void messageIdsReturnsDispatchThenEscalationsInSendOrder() {
            alert.recordDispatchMessage(501L);
            alert.recordEscalationMessage(502L);
            alert.recordEscalationMessage(503L);
            assertThat(alert.messageIds()).containsExactly(501L, 502L, 503L);
        }

        @Test
        void followUpIsNotPartOfTheEditTargetSet() {
            alert.recordDispatchMessage(501L);
            alert.recordFollowUpMessage(504L);
            assertThat(alert.followUpMessageId()).contains(504L);
            assertThat(alert.messageIds()).containsExactly(501L);
        }

        @Test
        void dispatchMessageCanOnlyBeRecordedOnce() {
            alert.recordDispatchMessage(501L);
            assertThatThrownBy(() -> alert.recordDispatchMessage(502L))
                    .isInstanceOf(IllegalStateException.class);
        }

        @Test
        void followUpMessageCanOnlyBeRecordedOnce() {
            alert.recordFollowUpMessage(504L);
            assertThatThrownBy(() -> alert.recordFollowUpMessage(505L))
                    .isInstanceOf(IllegalStateException.class);
        }

        @Test
        void escalationMessageIdsAreNotModifiableFromOutside() {
            alert.recordEscalationMessage(502L);
            assertThatThrownBy(() -> alert.escalationMessageIds().add(999L))
                    .isInstanceOf(UnsupportedOperationException.class);
        }
    }

    @Nested
    @DisplayName("clip attachment")
    class ClipAttachment {

        private static final Instant STORED_AT = Instant.parse("2026-09-16T10:05:00Z");

        @Test
        void newAlertHasNoClip() {
            assertThat(alert.clip()).isEmpty();
        }

        @Test
        void attachingAClipRecordsItAsStoredButNotDelivered() {
            alert.attachClip(SkeletonClip.stored("room-12/FE-1.mp4", STORED_AT));

            assertThat(alert.clip()).isPresent();
            assertThat(alert.clip().orElseThrow().storageKey()).isEqualTo("room-12/FE-1.mp4");
            assertThat(alert.clip().orElseThrow().isDelivered()).isFalse();
        }

        @Test
        void clipCanOnlyBeAttachedOnce() {
            alert.attachClip(SkeletonClip.stored("room-12/FE-1.mp4", STORED_AT));

            assertThatThrownBy(() -> alert.attachClip(SkeletonClip.stored("room-12/other.mp4", STORED_AT)))
                    .isInstanceOf(ClipAlreadyAttachedException.class);
        }

        @Test
        void recordingDeliveryKeepsTheStoredCopyAndAddsTheTelegramIds() {
            alert.attachClip(SkeletonClip.stored("room-12/FE-1.mp4", STORED_AT));

            alert.recordClipDelivery("file-abc", 601L);

            SkeletonClip clip = alert.clip().orElseThrow();
            assertThat(clip.storageKey()).isEqualTo("room-12/FE-1.mp4");
            assertThat(clip.storedAt()).isEqualTo(STORED_AT);
            assertThat(clip.telegramFileId()).isEqualTo("file-abc");
            assertThat(clip.telegramMessageId()).isEqualTo(601L);
            assertThat(clip.isDelivered()).isTrue();
        }

        @Test
        void deliveryCannotBeRecordedWithoutAClip() {
            assertThatThrownBy(() -> alert.recordClipDelivery("file-abc", 601L))
                    .isInstanceOf(IllegalStateException.class);
        }

        @Test
        void clipMessageIsNotPartOfTheEditTargetSet() {
            alert.recordDispatchMessage(501L);
            alert.attachClip(SkeletonClip.stored("room-12/FE-1.mp4", STORED_AT));
            alert.recordClipDelivery("file-abc", 601L);

            assertThat(alert.messageIds()).containsExactly(501L);
        }
    }

}
