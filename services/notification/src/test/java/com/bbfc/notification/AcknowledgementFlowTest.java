package com.bbfc.notification;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.containing;
import static com.github.tomakehurst.wiremock.client.WireMock.postRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo;
import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Import;
import org.springframework.context.annotation.Primary;
import org.springframework.http.MediaType;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.AlertState;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.Outcome;
import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.service.EscalationScheduler;
import com.bbfc.notification.out.persistence.AlertJPARepository;
import com.bbfc.notification.in.telegram.CallbackQueryHandler;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.CallbackQuery;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.Chat;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.Message;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.TelegramUser;
import com.bbfc.notification.testsupport.AbstractIntegrationTest;
import com.bbfc.notification.testsupport.MutableClock;
import com.github.tomakehurst.wiremock.client.WireMock;
import com.github.tomakehurst.wiremock.junit5.WireMockExtension;

/**
 * SPEC §12.2 end to end. The scheduler and the long poller are both off in the test profile and
 * driven directly here — @Scheduled and getUpdates are only triggers, and calling the methods
 * ourselves is what lets a fake clock stand in for minutes of waiting.
 */
@AutoConfigureMockMvc
@Import(AcknowledgementFlowTest.FakeClockConfig.class)
class AcknowledgementFlowTest extends AbstractIntegrationTest {

    private static final String EVENT_ID = "FE-20260916-0042";
    private static final Instant DETECTED = Instant.parse("2026-09-16T10:00:00Z");

    private static final String SEND = "/bottest-token/sendMessage";
    private static final String EDIT = "/bottest-token/editMessageText";
    private static final String ANSWER = "/bottest-token/answerCallbackQuery";

    @RegisterExtension
    static WireMockExtension telegram = WireMockExtension.newInstance().build();

    @DynamicPropertySource
    static void telegramBaseUrl(DynamicPropertyRegistry registry) {
        registry.add("telegram.base-url", telegram::baseUrl);
    }

    @TestConfiguration
    static class FakeClockConfig {
        @Bean
        @Primary
        Clock testClock() {
            return new MutableClock(DETECTED, ZoneOffset.UTC);
        }
    }

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private Clock clock;

    @Autowired
    private EscalationScheduler escalationScheduler;

    @Autowired
    private CallbackQueryHandler callbackQueryHandler;

    @Autowired
    private AlertRepository alertRepository;

    @Autowired
    private AlertJPARepository alertJpaRepository;

    @Autowired
    private TelegramConfig telegramConfig;

    @BeforeEach
    void resetState() {
        // @SpringBootTest does not roll back, so the alert would leak into the next method.
        alertJpaRepository.deleteAll();
        telegram.resetAll();
        stubSend(501L);
        telegram.stubFor(WireMock.post(urlEqualTo(EDIT)).willReturn(aResponse().withStatus(200)));
        telegram.stubFor(WireMock.post(urlEqualTo(ANSWER)).willReturn(aResponse().withStatus(200)));
    }

    private MutableClock mutableClock() {
        return (MutableClock) clock;
    }

    private void stubSend(long messageId) {
        telegram.stubFor(WireMock.post(urlEqualTo(SEND)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"ok\":true,\"result\":{\"message_id\":" + messageId + "}}")));
    }

    private void postEvent(int expectedStatus) throws Exception {
        mockMvc.perform(post("/events")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "eventId": "%s",
                                  "roomId": "room-12",
                                  "roomName": "Block A - Room 12",
                                  "confidence": 0.92
                                }""".formatted(EVENT_ID)))
                .andExpect(status().is(expectedStatus));
    }

    private void tap(String callbackData) {
        callbackQueryHandler.handle(new CallbackQuery(
                "cbq-1",
                new TelegramUser(777L, "nurse_lim", null, null),
                callbackData,
                new Message(new Chat(Long.parseLong(telegramConfig.chatGroupId())))));
    }

    private Alert reload() {
        return alertRepository.findByEventId(new EventId(EVENT_ID)).orElseThrow();
    }

    @Test
    void acknowledgementCancelsEscalationAndCapturesTheTriageOutcome() throws Exception {
        // 1. the alert goes out
        postEvent(202);
        telegram.verify(1, postRequestedFor(urlEqualTo(SEND)));

        // 2. nobody responds, so it escalates once
        stubSend(502L);
        mutableClock().advance(Duration.ofSeconds(61));
        escalationScheduler.escalateDueAlerts();
        telegram.verify(2, postRequestedFor(urlEqualTo(SEND)));
        assertThat(reload().state()).isEqualTo(AlertState.ESCALATING);

        // 3. a nurse taps Acknowledge: spinner cleared, both messages closed out, question asked
        stubSend(504L);
        tap("ack:" + EVENT_ID);

        telegram.verify(1, postRequestedFor(urlEqualTo(ANSWER)));
        telegram.verify(2, postRequestedFor(urlEqualTo(EDIT)));
        telegram.verify(postRequestedFor(urlEqualTo(EDIT))
                .withRequestBody(containing("\"inline_keyboard\":[]")));
        telegram.verify(3, postRequestedFor(urlEqualTo(SEND)));

        Alert acknowledged = reload();
        assertThat(acknowledged.state()).isEqualTo(AlertState.ACKNOWLEDGED);
        assertThat(acknowledged.acknowledgedBy()).contains("@nurse_lim");
        assertThat(acknowledged.nextEscalationAt()).isEmpty();
        assertThat(acknowledged.followUpMessageId()).contains(504L);

        // 4. the escalation is genuinely cancelled, not merely quiet
        mutableClock().advance(Duration.ofSeconds(61));
        escalationScheduler.escalateDueAlerts();
        telegram.verify(3, postRequestedFor(urlEqualTo(SEND)));
        telegram.verify(2, postRequestedFor(urlEqualTo(EDIT)));

        // 5. triage records the ground truth for the accuracy metric
        tap("triage:" + EVENT_ID + ":false_alarm");

        Alert triaged = reload();
        assertThat(triaged.state()).isEqualTo(AlertState.TRIAGED);
        assertThat(triaged.outcome()).contains(Outcome.FALSE_ALARM);
        assertThat(triaged.outcomeAt()).isPresent();
    }

    @Test
    void aDoubleTapChangesNothingAndTellsTheSecondResponderWhoHandledIt() throws Exception {
        postEvent(202);
        stubSend(504L);
        tap("ack:" + EVENT_ID);
        int sendsAfterFirstTap = telegram.findAll(postRequestedFor(urlEqualTo(SEND))).size();

        tap("ack:" + EVENT_ID);

        assertThat(reload().acknowledgedBy()).contains("@nurse_lim");
        telegram.verify(sendsAfterFirstTap, postRequestedFor(urlEqualTo(SEND)));
        telegram.verify(postRequestedFor(urlEqualTo(ANSWER))
                .withRequestBody(containing("Already handled by @nurse_lim")));
    }

    @Test
    void reIngestingTheSameEventIsADuplicateAndSendsNothing() throws Exception {
        postEvent(202);

        postEvent(200);

        telegram.verify(1, postRequestedFor(urlEqualTo(SEND)));
    }
}
