package com.bbfc.notification.in.telegram;

import java.time.Clock;
import java.time.Instant;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.core.domain.CallbackCommand;
import com.bbfc.notification.core.port.NotificationChannel;
import com.bbfc.notification.core.service.AcknowledgeAlertService;
import com.bbfc.notification.core.service.TriageAlertService;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.CallbackQuery;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.TelegramUser;

/** Turns a button tap into a use case, rejecting anything that didn't come from the ward group. */
@Component
public class CallbackQueryHandler {

    private static final Logger log = LoggerFactory.getLogger(CallbackQueryHandler.class);

    /** alerts.acknowledged_by is VARCHAR(64) and a Telegram display name can be longer. */
    private static final int MAX_RESPONDER_LENGTH = 64;

    private final AcknowledgeAlertService acknowledgeAlertService;
    private final TriageAlertService triageAlertService;
    private final NotificationChannel notificationChannel;
    private final String registeredChatId;
    private final Clock clock;

    public CallbackQueryHandler(
            AcknowledgeAlertService acknowledgeAlertService,
            TriageAlertService triageAlertService,
            NotificationChannel notificationChannel,
            TelegramConfig telegramConfig,
            Clock clock) {
        this.acknowledgeAlertService = acknowledgeAlertService;
        this.triageAlertService = triageAlertService;
        this.notificationChannel = notificationChannel;
        this.registeredChatId = telegramConfig.chatGroupId();
        this.clock = clock;
    }

    public void handle(CallbackQuery query) {
        String chatId = chatIdOf(query);
        if (chatId == null) {
            log.warn("Ignoring callback {} with no accessible chat", query.id());
            return;
        }

        if (!registeredChatId.equals(chatId)) {
            log.warn("Rejected callback from unregistered chat {}", chatId);
            notificationChannel.answerCallback(query.id(), "This bot only answers the registered ward group.");
            return;
        }

        CallbackCommand command;
        try {
            command = CallbackCommand.parse(query.data());
        } catch (IllegalArgumentException e) {
            log.warn("Rejected unparseable callback_data from chat {}: {}", chatId, e.getMessage());
            notificationChannel.answerCallback(query.id(), "That button is no longer valid.");
            return;
        }

        Instant now = clock.instant();
        String responder = responderName(query.from());
        switch (command) {
            case CallbackCommand.Acknowledge ack ->
                    acknowledgeAlertService.acknowledge(query.id(), ack.eventId(), responder, now);
            case CallbackCommand.Triage triage ->
                    triageAlertService.triage(query.id(), triage.eventId(), triage.outcome(), now);
        }
    }

    private static String chatIdOf(CallbackQuery query) {
        if (query.message() == null || query.message().chat() == null) {
            return null;
        }
        return String.valueOf(query.message().chat().id());
    }

    private static String responderName(TelegramUser user) {
        if (user == null) {
            return "unknown";
        }
        if (user.username() != null && !user.username().isBlank()) {
            return truncate("@" + user.username());
        }
        String fullName = (orEmpty(user.firstName()) + " " + orEmpty(user.lastName())).trim();
        return fullName.isEmpty() ? "user " + user.id() : truncate(fullName);
    }

    private static String orEmpty(String value) {
        return value == null ? "" : value;
    }

    private static String truncate(String value) {
        return value.length() <= MAX_RESPONDER_LENGTH ? value : value.substring(0, MAX_RESPONDER_LENGTH);
    }
}
