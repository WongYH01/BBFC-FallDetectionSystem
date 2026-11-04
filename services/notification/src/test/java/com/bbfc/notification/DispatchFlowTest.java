package com.bbfc.notification;

import static com.github.tomakehurst.wiremock.client.WireMock.aMultipart;
import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.anyUrl;
import static com.github.tomakehurst.wiremock.client.WireMock.binaryEqualTo;
import static com.github.tomakehurst.wiremock.client.WireMock.containing;
import static com.github.tomakehurst.wiremock.client.WireMock.equalTo;
import static com.github.tomakehurst.wiremock.client.WireMock.matchingJsonPath;
import static com.github.tomakehurst.wiremock.client.WireMock.postRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo;
import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.nio.charset.StandardCharsets;
import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Import;
import org.springframework.context.annotation.Primary;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.AlertState;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.RoomRef;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.service.AlertDispatcher;
import com.bbfc.notification.core.service.EscalationScheduler;
import com.bbfc.notification.in.telegram.CallbackQueryHandler;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.CallbackQuery;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.Chat;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.Message;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.TelegramUser;
import com.bbfc.notification.out.persistence.AlertJPARepository;
import com.bbfc.notification.testsupport.AbstractIntegrationTest;
import com.bbfc.notification.testsupport.MutableClock;
import com.github.tomakehurst.wiremock.client.WireMock;
import com.github.tomakehurst.wiremock.junit5.WireMockExtension;

/** An Alert is held as Pending for its clip, then dispatched exactly once. */
@AutoConfigureMockMvc
@Import(DispatchFlowTest.FakeClockConfig.class)
class DispatchFlowTest extends AbstractIntegrationTest {

    private static final String EVENT_ID = "FE-20260916-0099";
    // 6:00:50pm in Singapore, so a dispatch at the 20s deadline falls in the next minute.
    private static final Instant RECEIVED = Instant.parse("2026-09-16T10:00:50Z");
    private static final String SEND = "/bottest-token/sendMessage";
    private static final String ANIMATION = "/bottest-token/sendAnimation";
    private static final String EDIT_TEXT = "/bottest-token/editMessageText";
    private static final String EDIT_CAPTION = "/bottest-token/editMessageCaption";
    private static final String ANSWER = "/bottest-token/answerCallbackQuery";
    private static final String OBJECT = "/object/skeleton-clips/room-12/" + EVENT_ID + ".mp4";
    private static final byte[] CLIP_BYTES = "fake-skeleton-mp4-bytes".getBytes(StandardCharsets.UTF_8);

    @RegisterExtension
    static WireMockExtension telegram = WireMockExtension.newInstance().build();

    @RegisterExtension
    static WireMockExtension storage = WireMockExtension.newInstance().build();

    @DynamicPropertySource
    static void baseUrls(DynamicPropertyRegistry registry) {
        registry.add("telegram.base-url", telegram::baseUrl);
        registry.add("supabase.storage.url", storage::baseUrl);
    }

    @TestConfiguration
    static class FakeClockConfig {
        @Bean
        @Primary
        Clock testClock() {
            return new MutableClock(RECEIVED, ZoneOffset.UTC);
        }
    }

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private Clock clock;

    @Autowired
    private AlertDispatcher alertDispatcher;

    @Autowired
    private EscalationScheduler escalationScheduler;

    @Autowired
    private CallbackQueryHandler callbackQueryHandler;

    @Autowired
    private TelegramConfig telegramConfig;

    @Autowired
    private AlertRepository alertRepository;

    @Autowired
    private AlertJPARepository alertJpaRepository;

    @BeforeEach
    void resetState() {
        alertJpaRepository.deleteAll();
        telegram.resetAll();
        storage.resetAll();
        ((MutableClock) clock).set(RECEIVED);
        stubSend(0);
        stubAnimation(0);
        telegram.stubFor(WireMock.post(urlEqualTo(EDIT_TEXT)).willReturn(aResponse().withStatus(200)));
        telegram.stubFor(WireMock.post(urlEqualTo(EDIT_CAPTION)).willReturn(aResponse().withStatus(200)));
        telegram.stubFor(WireMock.post(urlEqualTo(ANSWER)).willReturn(aResponse().withStatus(200)));
        storage.stubFor(WireMock.put(urlEqualTo(OBJECT)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"Key\":\"skeleton-clips/room-12/" + EVENT_ID + ".mp4\"}")));
    }

    private void stubSend(int delayMillis) {
        telegram.stubFor(WireMock.post(urlEqualTo(SEND)).willReturn(aResponse()
                .withStatus(200)
                .withFixedDelay(delayMillis)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"ok\":true,\"result\":{\"message_id\":501}}")));
    }

    private void stubAnimation(int delayMillis) {
        telegram.stubFor(WireMock.post(urlEqualTo(ANIMATION)).willReturn(aResponse()
                .withStatus(200)
                .withFixedDelay(delayMillis)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"ok\":true,\"result\":{\"message_id\":601,"
                        + "\"animation\":{\"file_id\":\"file-abc\"}}}")));
    }

    private void advance(Duration duration) {
        ((MutableClock) clock).advance(duration);
    }

    private void postEvent() throws Exception {
        mockMvc.perform(post("/events")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "eventId": "%s",
                                  "roomId": "room-12",
                                  "roomName": "Block A - Room 12",
                                  "confidence": 0.92
                                }""".formatted(EVENT_ID)))
                .andExpect(status().isAccepted());
    }

    private ResultActions uploadClip() throws Exception {
        MockMultipartFile file = new MockMultipartFile("file", "pose_skeleton.mp4", "video/mp4", CLIP_BYTES);
        return mockMvc.perform(multipart(HttpMethod.PUT, "/events/{id}/clip", EVENT_ID).file(file));
    }

    private void tap() {
        callbackQueryHandler.handle(new CallbackQuery(
                "cbq-1",
                new TelegramUser(777L, "nurse_lim", null, null),
                "ack:" + EVENT_ID,
                new Message(new Chat(Long.parseLong(telegramConfig.chatGroupId())))));
    }

    private void passTheDeadlineAndTick() {
        advance(Duration.ofSeconds(20));
        alertDispatcher.dispatchDueAlerts();
    }

    private Alert reload() {
        return alertRepository.findByEventId(new EventId(EVENT_ID)).orElseThrow();
    }

    @Test
    void receivingAnEventSendsNothingAndLeavesTheAlertPending() throws Exception {
        postEvent();

        telegram.verify(0, postRequestedFor(anyUrl()));
        assertThat(reload().state()).isEqualTo(AlertState.PENDING);
    }

    @Test
    void anEventReSentDuringTheHoldIsADuplicateAndIsDispatchedOnce() throws Exception {
        postEvent();
        advance(Duration.ofSeconds(10));

        mockMvc.perform(post("/events")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "eventId": "%s",
                                  "roomId": "room-12",
                                  "roomName": "Block A - Room 12",
                                  "confidence": 0.92
                                }""".formatted(EVENT_ID)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("duplicate"));
        advance(Duration.ofSeconds(10));
        alertDispatcher.dispatchDueAlerts();

        telegram.verify(1, postRequestedFor(urlEqualTo(SEND)));
    }

    @Test
    void withNoClipTheTickSendsATextAlertOnceTheDeadlinePassesAndNotBefore() throws Exception {
        postEvent();

        advance(Duration.ofSeconds(19));
        alertDispatcher.dispatchDueAlerts();
        telegram.verify(0, postRequestedFor(anyUrl()));

        advance(Duration.ofSeconds(1));
        alertDispatcher.dispatchDueAlerts();
        alertDispatcher.dispatchDueAlerts();

        telegram.verify(1, postRequestedFor(urlEqualTo(SEND)));
        telegram.verify(postRequestedFor(urlEqualTo(SEND))
                .withRequestBody(matchingJsonPath("$.reply_markup.inline_keyboard[0][0].callback_data",
                        equalTo("ack:" + EVENT_ID))));
        Alert dispatched = reload();
        assertThat(dispatched.state()).isEqualTo(AlertState.DISPATCHED);
        assertThat(dispatched.dispatchMessageId()).contains(501L);
        assertThat(dispatched.dispatchCarriesClip()).isFalse();
    }

    @Test
    void theAlertShowsWhenItWasReceivedNotWhenItWasDispatched() throws Exception {
        postEvent();

        passTheDeadlineAndTick();

        telegram.verify(postRequestedFor(urlEqualTo(SEND))
                .withRequestBody(matchingJsonPath("$.text", containing("16 September 2026 6:00pm"))));
    }

    @Test
    void aClipWithinTheWaitIsDispatchedAsOneCaptionedAnimation() throws Exception {
        postEvent();
        advance(Duration.ofSeconds(10));

        uploadClip().andExpect(status().isAccepted()).andExpect(jsonPath("$.sentToTelegram").value(true));
        passTheDeadlineAndTick();

        telegram.verify(1, postRequestedFor(urlEqualTo(ANIMATION)));
        telegram.verify(postRequestedFor(urlEqualTo(ANIMATION))
                .withRequestBodyPart(aMultipart().withName("caption")
                        .withBody(containing("Block A - Room 12")).build())
                .withRequestBodyPart(aMultipart().withName("parse_mode").withBody(equalTo("HTML")).build())
                .withRequestBodyPart(aMultipart().withName("reply_markup")
                        .withBody(containing("ack:" + EVENT_ID)).build())
                .withRequestBodyPart(aMultipart().withName("animation")
                        .withBody(binaryEqualTo(CLIP_BYTES)).build()));
        telegram.verify(0, postRequestedFor(urlEqualTo(SEND)));

        Alert dispatched = reload();
        assertThat(dispatched.state()).isEqualTo(AlertState.DISPATCHED);
        assertThat(dispatched.dispatchMessageId()).contains(601L);
        assertThat(dispatched.dispatchCarriesClip()).isTrue();
        assertThat(dispatched.clip().orElseThrow().telegramFileId()).isEqualTo("file-abc");
    }

    @Test
    void aPendingAlertLeftOverFromBeforeARestartIsDispatchedByTheFirstTick() {
        alertRepository.save(Alert.receive(
                new EventId(EVENT_ID), new RoomRef("room-12", "Block A - Room 12"), new Confidence(0.92),
                RECEIVED.minus(Duration.ofMinutes(5)), RECEIVED.minus(Duration.ofMinutes(5)).plusSeconds(20)));

        alertDispatcher.dispatchDueAlerts();

        telegram.verify(1, postRequestedFor(urlEqualTo(SEND)));
        assertThat(reload().state()).isEqualTo(AlertState.DISPATCHED);
    }

    @Test
    void aTickRacingAClipThatHoldsTheAlertDoesNotDispatchItAgain() throws Exception {
        postEvent();
        advance(Duration.ofSeconds(20));
        stubAnimation(1500);

        CompletableFuture<Void> clip = CompletableFuture.runAsync(() -> {
            try {
                uploadClip().andExpect(status().isAccepted());
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
        });
        awaitRequest(ANIMATION);
        alertDispatcher.dispatchDueAlerts();
        clip.get(10, TimeUnit.SECONDS);
        alertDispatcher.dispatchDueAlerts();

        telegram.verify(0, postRequestedFor(urlEqualTo(SEND)));
        telegram.verify(1, postRequestedFor(urlEqualTo(ANIMATION)));
        assertThat(reload().dispatchCarriesClip()).isTrue();
    }

    @Test
    void aClipRacingATickThatHoldsTheAlertDoesNotDispatchItAgain() throws Exception {
        postEvent();
        advance(Duration.ofSeconds(20));
        stubSend(1500);

        CompletableFuture<Void> tick = CompletableFuture.runAsync(alertDispatcher::dispatchDueAlerts);
        awaitRequest(SEND);
        uploadClip().andExpect(status().isAccepted());
        tick.get(10, TimeUnit.SECONDS);

        telegram.verify(1, postRequestedFor(urlEqualTo(SEND)));
        // Until #49 the late clip still goes out, but as a reply to the text dispatch, not a second dispatch.
        telegram.verify(1, postRequestedFor(urlEqualTo(ANIMATION))
                .withRequestBodyPart(aMultipart().withName("reply_to_message_id").withBody(equalTo("501")).build()));
        Alert dispatched = reload();
        assertThat(dispatched.dispatchMessageId()).contains(501L);
        assertThat(dispatched.dispatchCarriesClip()).isFalse();
    }

    @Test
    void theFirstEscalationComesOneWindowAfterDispatchNotAfterReceipt() throws Exception {
        postEvent();
        passTheDeadlineAndTick();

        advance(Duration.ofSeconds(59));
        escalationScheduler.escalateDueAlerts();
        telegram.verify(1, postRequestedFor(urlEqualTo(SEND)));

        advance(Duration.ofSeconds(1));
        escalationScheduler.escalateDueAlerts();
        telegram.verify(2, postRequestedFor(urlEqualTo(SEND)));
        assertThat(reload().state()).isEqualTo(AlertState.ESCALATING);
    }

    @Test
    void aCallbackForAPendingAlertDoesNotAcknowledgeIt() throws Exception {
        postEvent();

        tap();

        telegram.verify(1, postRequestedFor(urlEqualTo(ANSWER)));
        Alert pending = reload();
        assertThat(pending.state()).isEqualTo(AlertState.PENDING);
        assertThat(pending.acknowledgedBy()).isEmpty();
    }

    @Test
    void acknowledgingAClipDispatchEditsItsCaption() throws Exception {
        postEvent();
        uploadClip().andExpect(status().isAccepted());

        tap();

        telegram.verify(postRequestedFor(urlEqualTo(EDIT_CAPTION))
                .withRequestBody(matchingJsonPath("$.message_id", equalTo("601")))
                .withRequestBody(containing("\"inline_keyboard\":[]")));
        telegram.verify(0, postRequestedFor(urlEqualTo(EDIT_TEXT)));
    }

    @Test
    void acknowledgingATextDispatchEditsItsText() throws Exception {
        postEvent();
        passTheDeadlineAndTick();

        tap();

        telegram.verify(postRequestedFor(urlEqualTo(EDIT_TEXT))
                .withRequestBody(matchingJsonPath("$.message_id", equalTo("501")))
                .withRequestBody(containing("\"inline_keyboard\":[]")));
        telegram.verify(0, postRequestedFor(urlEqualTo(EDIT_CAPTION)));
    }

    private static void awaitRequest(String url) throws InterruptedException {
        for (int i = 0; i < 100 && telegram.findAll(postRequestedFor(urlEqualTo(url))).isEmpty(); i++) {
            Thread.sleep(50);
        }
        assertThat(telegram.findAll(postRequestedFor(urlEqualTo(url)))).isNotEmpty();
    }
}
