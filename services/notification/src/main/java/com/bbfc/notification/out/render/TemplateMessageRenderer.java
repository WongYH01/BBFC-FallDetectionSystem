package com.bbfc.notification.out.render;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.Outcome;
import com.bbfc.notification.core.port.MessageRenderer;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.util.HtmlUtils;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.temporal.ChronoField;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Component
public class TemplateMessageRenderer implements MessageRenderer {

    private static  final Pattern PLACEHOLDER = Pattern.compile("\\{(\\w+)}");
    private static  final Set<String> KNOWN_VALUES = Set.of("roomName","time","confidence","eventId");
    private static final Set<String> TRIAGE_QUESTION_KNOWN_VALUES = Set.of("roomName","eventId","handledLine");

    private final String messageTemplate;
    private final String triageQuestionTemplate;

    private static final DateTimeFormatter TIME_FORMAT = new DateTimeFormatterBuilder()
            .appendPattern("d MMMM yyyy h:mm")
            .appendText(ChronoField.AMPM_OF_DAY, Map.of(0L, "am", 1L, "pm"))
            .toFormatter()
            .withZone(ZoneId.systemDefault());

    private static final DateTimeFormatter CLOCK_TIME = DateTimeFormatter.ofPattern("HH:mm")
            .withZone(ZoneId.systemDefault());

    private static void validateTemplate(String template, Set<String> knownValues, String propertyName){
        Matcher matcher = PLACEHOLDER.matcher(template);
        while (matcher.find()){
            String name = matcher.group(1);
            if (!knownValues.contains(name)){
                throw new IllegalStateException(
                        "telegram."+propertyName+" references unknown placeholder {"+name+"}"
                );
            }
        }
    }

    public TemplateMessageRenderer(TelegramConfig telegramConfig){
        this.messageTemplate = telegramConfig.messageTemplate();
        validateTemplate(messageTemplate, KNOWN_VALUES, "message-template");
        this.triageQuestionTemplate = telegramConfig.triageQuestionTemplate();
        validateTemplate(triageQuestionTemplate, TRIAGE_QUESTION_KNOWN_VALUES, "triage-question-template");
    }


    @Override
    public String render(Alert alert, Instant at){
        String roomName = HtmlUtils.htmlEscape(alert.room().displayName());

        String time = HtmlUtils.htmlEscape(TIME_FORMAT.format(at));
        String confidence = HtmlUtils.htmlEscape(
                StringUtils.capitalize(alert.confidence().band().name().toLowerCase())
                        + " (" + alert.confidence().confidenceScore() + ")");
        String eventId = HtmlUtils.htmlEscape(alert.eventId().eventId());

        return messageTemplate
                .replace("{roomName}", roomName)
                .replace("{time}", time)
                .replace("{confidence}", confidence)
                .replace("{eventId}", eventId);
    }

    @Override
    public String renderAcknowledged(Alert alert) {
        return """
                ✅ <b>FALL ALERT — HANDLED</b>
                <b>Room</b> %s
                <b>Detected</b> %s · <b>Handled</b> %s by %s
                <i>Ref %s</i>"""
                .formatted(
                        HtmlUtils.htmlEscape(alert.room().displayName()),
                        CLOCK_TIME.format(alert.createdAt()),
                        alert.acknowledgedAt().map(CLOCK_TIME::format).orElse("—"),
                        responder(alert),
                        HtmlUtils.htmlEscape(alert.eventId().eventId()));
    }

    @Override
    public String renderTriageQuestion(Alert alert) {
        return triageQuestionTemplate
                .replace("{eventId}", HtmlUtils.htmlEscape(alert.eventId().eventId()))
                .replace("{roomName}", HtmlUtils.htmlEscape(alert.room().displayName()))
                .replace("{handledLine}", handledLine(alert));
    }

    @Override
    public String renderTriaged(Alert alert) {
        String outcome = alert.outcome()
                .map(o -> o == Outcome.GENUINE ? "👍 Recorded: Genuine fall" : "🚫 Recorded: False alarm")
                .orElse("Recorded");
        return """
                <b>Ref %s</b> · %s
                %s
                %s"""
                .formatted(
                        HtmlUtils.htmlEscape(alert.eventId().eventId()),
                        HtmlUtils.htmlEscape(alert.room().displayName()),
                        handledLine(alert),
                        outcome);
    }

    private static String handledLine(Alert alert) {
        return "Fall detected %s, acknowledged %s by %s.".formatted(
                CLOCK_TIME.format(alert.createdAt()),
                alert.acknowledgedAt().map(CLOCK_TIME::format).orElse("—"),
                responder(alert));
    }

    // Comes from a Telegram display name, so it is attacker-controlled and lands in a parse_mode=HTML body.
    private static String responder(Alert alert) {
        return HtmlUtils.htmlEscape(alert.acknowledgedBy().orElse("someone"));
    }
}
