package com.bbfc.notification.out.render;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.Outcome;
import com.bbfc.notification.core.domain.RoomRef;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TemplateMessageRendererTest {
    private static TelegramConfig properties(String template) {
        return new TelegramConfig("token", "chat-id", "http://localhost", template,
                "<b>Ref {eventId}</b> · {roomName}\n{handledLine}\n<b>Was this a genuine fall?</b>");
    }

    @Test
    void successSubstituteKnownValues(){
        TemplateMessageRenderer templateMessageRenderer = new TemplateMessageRenderer(
                properties("Fall in {roomName} at {time}, confidence {confidence}")
        );
        Alert alert = Alert.dispatch(
                new EventId("FE-1"), new RoomRef("room-12", "Block A - Room 12"), new Confidence(0.9),
                Instant.parse("2026-09-16T10:00:00Z"));

        String rendered = templateMessageRenderer.render(alert, Instant.parse("2026-09-16T10:00:00Z"));

        assertThat(rendered).isEqualTo("Fall in Block A - Room 12 at 16 September 2026 6:00pm, confidence High (0.9)");
    }

    @Test
    void checkXSSDefense(){
        TemplateMessageRenderer renderer = new TemplateMessageRenderer(properties("{roomName}"));
        Alert alert = Alert.dispatch(
                new EventId("FE-1"), new RoomRef("room-12", "<script>alert(1)</script>"), new Confidence(0.67),
                Instant.now());

        String rendered = renderer.render(alert, Instant.now());

        assertThat(rendered).doesNotContain("<script>");
        assertThat(rendered).contains("&lt;script&gt;");
    }

    private static final DateTimeFormatter CLOCK_TIME =
            DateTimeFormatter.ofPattern("HH:mm").withZone(ZoneId.systemDefault());

    private static final Instant DETECTED = Instant.parse("2026-09-16T10:00:00Z");
    private static final Instant HANDLED = Instant.parse("2026-09-16T10:03:00Z");

    private static Alert acknowledgedAlert(String responder) {
        Alert alert = Alert.dispatch(
                new EventId("FE-20260916-0003"),
                new RoomRef("room-12", "Block A - Room 12"),
                new Confidence(0.92),
                DETECTED);
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
                .contains("FE-20260916-0003")
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
                .contains("FE-20260916-0003")
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

    @Test
    void checkRejectUnknownValue(){
        assertThatThrownBy(() -> new TemplateMessageRenderer(properties("Fall in {residentName}")))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("residentName");
    }
}
