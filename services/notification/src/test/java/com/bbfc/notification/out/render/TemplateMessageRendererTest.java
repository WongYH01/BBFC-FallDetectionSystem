package com.bbfc.notification.out.render;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.RoomRef;
import org.junit.jupiter.api.Test;

import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TemplateMessageRendererTest {
    private static TelegramConfig properties(String template) {
        return new TelegramConfig("token", "chat-id", "http://localhost", template);
    }

    @Test
    void successSubstituteKnownValues(){
        TemplateMessageRenderer templateMessageRenderer = new TemplateMessageRenderer(
                properties("Fall in {roomName} at {time}, confidence {confidence}")
        );
        Alert alert = Alert.dispatch(new EventId("FE-1"), new RoomRef("room-12", "Block A - Room 12"), new Confidence(0.9));

        String rendered = templateMessageRenderer.render(alert, Instant.parse("2026-09-16T10:00:00Z"));

        assertThat(rendered).isEqualTo("Fall in Block A - Room 12 at 16 September 2026 6:00pm, confidence High (0.9)");
    }

    @Test
    void checkXSSDefense(){
        TemplateMessageRenderer renderer = new TemplateMessageRenderer(properties("{roomName}"));
        Alert alert = Alert.dispatch(
                new EventId("FE-1"), new RoomRef("room-12", "<script>alert(1)</script>"), new Confidence(0.67));

        String rendered = renderer.render(alert, Instant.now());

        assertThat(rendered).doesNotContain("<script>");
        assertThat(rendered).contains("&lt;script&gt;");
    }

    @Test
    void checkRejectUnknownValue(){
        assertThatThrownBy(() -> new TemplateMessageRenderer(properties("Fall in {residentName}")))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("residentName");
    }
}
