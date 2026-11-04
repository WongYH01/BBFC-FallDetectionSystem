package com.bbfc.notification.out.render;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.Outcome;
import com.bbfc.notification.core.domain.RoomRef;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TemplateMessageRendererTest {
    private static TelegramConfig properties(String template) {
        return new TelegramConfig("token", "chat-id", "http://localhost", template,
                "<b>{roomName}</b>\n{handledLine}\n<b>Was this a genuine fall?</b>",
                "⚠️ <b>{ordinal} escalation</b> · {roomName} · not yet acknowledged");
    }

    @Test
    void successSubstituteKnownValues(){
        TemplateMessageRenderer templateMessageRenderer = new TemplateMessageRenderer(
                properties("Fall in {roomName} at {time}, confidence {confidence}")
        );
        Alert alert = Alert.receive(
                new EventId("FE-1"), new RoomRef("room-12", "Block A - Room 12"), new Confidence(0.9),
                Instant.parse("2026-09-16T10:00:00Z"), Instant.parse("2026-09-16T10:00:00Z"));

        String rendered = templateMessageRenderer.render(alert, Instant.parse("2026-09-16T10:00:00Z"));

        assertThat(rendered).isEqualTo("Fall in Block A - Room 12 at 16 September 2026 6:00pm, confidence High (0.9)");
    }

    @Test
    void checkXSSDefense(){
        TemplateMessageRenderer renderer = new TemplateMessageRenderer(properties("{roomName}"));
        Alert alert = Alert.receive(
                new EventId("FE-1"), new RoomRef("room-12", "<script>alert(1)</script>"), new Confidence(0.67),
                Instant.now(), Instant.now());

        String rendered = renderer.render(alert, Instant.now());

        assertThat(rendered).doesNotContain("<script>");
        assertThat(rendered).contains("&lt;script&gt;");
    }

    private static final DateTimeFormatter CLOCK_TIME =
            DateTimeFormatter.ofPattern("HH:mm").withZone(ZoneId.of("Asia/Singapore"));

    private static final Instant DETECTED = Instant.parse("2026-09-16T10:00:00Z");
    private static final Instant HANDLED = Instant.parse("2026-09-16T10:03:00Z");

    private static Alert acknowledgedAlert(String responder) {
        Alert alert = Alert.receive(
                new EventId("FE-20260916-0003"),
                new RoomRef("room-12", "Block A - Room 12"),
                new Confidence(0.92),
                DETECTED, DETECTED);
        alert.dispatch(501L, false, null);
        alert.acknowledge(responder, HANDLED);
        return alert;
    }

    @Test
    void renderAcknowledgedShowsBothDetectionAndHandlingTimes() {
        TemplateMessageRenderer renderer = new TemplateMessageRenderer(properties("{roomName}"));

        String rendered = renderer.renderAcknowledged(acknowledgedAlert("@nurse_lim"));

        assertThat(rendered)
                .contains("HANDLED")
                .contains("Block A - Room 12")
                .contains("@nurse_lim")
                .contains(CLOCK_TIME.format(DETECTED))
                .contains(CLOCK_TIME.format(HANDLED));
    }

    @Test
    void renderTriageQuestionAsksTheQuestion() {
        TemplateMessageRenderer renderer = new TemplateMessageRenderer(properties("{roomName}"));

        String rendered = renderer.renderTriageQuestion(acknowledgedAlert("@nurse_lim"));

        assertThat(rendered)
                .contains("Was this a genuine fall?")
                .contains("@nurse_lim");
    }

    @Test
    void renderTriagedShowsTheRecordedOutcome() {
        TemplateMessageRenderer renderer = new TemplateMessageRenderer(properties("{roomName}"));
        Alert alert = acknowledgedAlert("@nurse_lim");
        alert.triage(Outcome.FALSE_ALARM, HANDLED);

        String rendered = renderer.renderTriaged(alert);

        assertThat(rendered).contains("False alarm").doesNotContain("Was this a genuine fall?");
    }

    @Test
    void escapesTheResponderNameInEveryFollowUpMessage() {
        TemplateMessageRenderer renderer = new TemplateMessageRenderer(properties("{roomName}"));
        Alert alert = acknowledgedAlert("<script>alert(1)</script>");

        assertThat(renderer.renderAcknowledged(alert)).doesNotContain("<script>").contains("&lt;script&gt;");
        assertThat(renderer.renderTriageQuestion(alert)).doesNotContain("<script>").contains("&lt;script&gt;");
    }

    @ParameterizedTest
    @CsvSource({"1,1st", "2,2nd", "3,3rd", "4,4th", "11,11th", "12,12th", "13,13th", "21,21st", "22,22nd", "111,111th"})
    void renderEscalationNumbersTheEscalationInEnglish(int escalations, String ordinal) {
        TemplateMessageRenderer renderer = new TemplateMessageRenderer(properties("{roomName}"));
        Alert alert = Alert.receive(
                new EventId("FE-1"), new RoomRef("room-12", "Block A - Room 12"), new Confidence(0.9), DETECTED, DETECTED);
        alert.dispatch(501L, false, null);
        for (int i = 0; i < escalations; i++) {
            alert.escalate(DETECTED);
        }

        assertThat(renderer.renderEscalation(alert))
                .isEqualTo("⚠️ <b>" + ordinal + " escalation</b> · Block A - Room 12 · not yet acknowledged");
    }

    @Test
    void eventIdIsNoLongerAPlaceholder() {
        assertThatThrownBy(() -> new TemplateMessageRenderer(properties("{roomName} Ref {eventId}")))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("message-template")
                .hasMessageContaining("eventId");
        assertThatThrownBy(() -> new TemplateMessageRenderer(new TelegramConfig(
                "token", "chat-id", "http://localhost", "{roomName}", "Ref {eventId}", "{ordinal}")))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("triage-question-template");
    }

    @Test
    void rejectsAnUnknownPlaceholderInTheEscalationTemplate() {
        assertThatThrownBy(() -> new TemplateMessageRenderer(new TelegramConfig(
                "token", "chat-id", "http://localhost", "{roomName}", "{roomName}", "{ordinal} {time}")))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("escalation-template")
                .hasMessageContaining("time");
    }

    @Test
    void checkRejectUnknownValue(){
        assertThatThrownBy(() -> new TemplateMessageRenderer(properties("Fall in {residentName}")))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("residentName");
    }
}
