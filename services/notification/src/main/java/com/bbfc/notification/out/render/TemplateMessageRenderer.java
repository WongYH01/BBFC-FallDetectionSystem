package com.bbfc.notification.out.render;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.core.domain.Alert;
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

    private final String messageTemplate;

    private static final DateTimeFormatter TIME_FORMAT = new DateTimeFormatterBuilder()
            .appendPattern("d MMMM yyyy h:mm")
            .appendText(ChronoField.AMPM_OF_DAY, Map.of(0L, "am", 1L, "pm"))
            .toFormatter()
            .withZone(ZoneId.systemDefault());

    private  static  void validateTemplate(String template){
        Matcher matcher = PLACEHOLDER.matcher(template);
        while (matcher.find()){
            String name = matcher.group(1);
            if (!KNOWN_VALUES.contains(name)){
                throw new IllegalStateException(
                        "telegram.message-template references unknown placeholder {"+name+"}"
                );
            }
        }
    }

    public TemplateMessageRenderer(TelegramConfig telegramConfig){
        this.messageTemplate = telegramConfig.messageTemplate();
        validateTemplate(messageTemplate);
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
}
