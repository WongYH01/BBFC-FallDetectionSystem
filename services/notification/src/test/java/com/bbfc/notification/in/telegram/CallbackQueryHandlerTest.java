package com.bbfc.notification.in.telegram;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.Outcome;
import com.bbfc.notification.core.port.NotificationChannel;
import com.bbfc.notification.core.service.AcknowledgeAlertService;
import com.bbfc.notification.core.service.TriageAlertService;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.CallbackQuery;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.Chat;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.Message;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.TelegramUser;

@ExtendWith(MockitoExtension.class)
class CallbackQueryHandlerTest {

    private static final long WARD_GROUP = -100123L;
    private static final Instant NOW = Instant.parse("2026-09-16T10:03:00Z");

    @Mock
    private AcknowledgeAlertService acknowledgeAlertService;

    @Mock
    private TriageAlertService triageAlertService;

    @Mock
    private NotificationChannel notificationChannel;

    private CallbackQueryHandler handler;

    @BeforeEach
    void setUp() {
        TelegramConfig config = new TelegramConfig(
                "test-token", String.valueOf(WARD_GROUP), "http://localhost", "{roomName}", "{roomName}");
        handler = new CallbackQueryHandler(
                acknowledgeAlertService, triageAlertService, notificationChannel, config,
                Clock.fixed(NOW, ZoneOffset.UTC));
    }

    private static CallbackQuery callback(long chatId, TelegramUser from, String data) {
        return new CallbackQuery("cbq-1", from, data, new Message(new Chat(chatId)));
    }

    private static TelegramUser user(String username, String firstName, String lastName) {
        return new TelegramUser(777L, username, firstName, lastName);
    }

    @Test
    void routesAcknowledgeToTheAcknowledgeService() {
        handler.handle(callback(WARD_GROUP, user("nurse_lim", null, null), "ack:FE-1"));

        verify(acknowledgeAlertService).acknowledge("cbq-1", new EventId("FE-1"), "@nurse_lim", NOW);
        verify(triageAlertService, never()).triage(any(), any(), any(), any());
    }

    @Test
    void routesTriageToTheTriageService() {
        handler.handle(callback(WARD_GROUP, user("nurse_lim", null, null), "triage:FE-1:false_alarm"));

        verify(triageAlertService).triage("cbq-1", new EventId("FE-1"), Outcome.FALSE_ALARM, NOW);
        verify(acknowledgeAlertService, never()).acknowledge(any(), any(), any(), any());
    }

    @Test
    void rejectsCallbacksFromUnregisteredChats() {
        handler.handle(callback(-999L, user("stranger", null, null), "ack:FE-1"));

        verify(notificationChannel).answerCallback(eq("cbq-1"), any());
        verify(acknowledgeAlertService, never()).acknowledge(any(), any(), any(), any());
    }

    @Test
    void rejectsUnparseableCallbackData() {
        handler.handle(callback(WARD_GROUP, user("nurse_lim", null, null), "gibberish"));

        verify(notificationChannel).answerCallback("cbq-1", "That button is no longer valid.");
        verify(acknowledgeAlertService, never()).acknowledge(any(), any(), any(), any());
    }

    @Test
    void ignoresCallbacksWithNoAccessibleMessage() {
        handler.handle(new CallbackQuery("cbq-1", user("nurse_lim", null, null), "ack:FE-1", null));

        verify(acknowledgeAlertService, never()).acknowledge(any(), any(), any(), any());
        verify(notificationChannel, never()).answerCallback(any(), any());
    }

    @ParameterizedTest
    @CsvSource({
            "nurse_lim, Amy,  Lim,  @nurse_lim",
            "          , Amy,  Lim,  Amy Lim",
            "          , Amy,      , Amy",
            "          ,     ,      , user 777"
    })
    void derivesTheResponderNameWithFallbacks(String username, String first, String last, String expected) {
        handler.handle(callback(WARD_GROUP, user(username, first, last), "ack:FE-1"));

        verify(acknowledgeAlertService).acknowledge(any(), any(), eq(expected), any());
    }

    @Test
    void truncatesAnOverlongResponderNameToFitTheColumn() {
        handler.handle(callback(WARD_GROUP, user("x".repeat(100), null, null), "ack:FE-1"));

        verify(acknowledgeAlertService).acknowledge(any(), any(), eq("@" + "x".repeat(63)), any());
    }
}
