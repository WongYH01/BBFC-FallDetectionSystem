package com.bbfc.notification.out.telegram;

import static com.github.tomakehurst.wiremock.client.WireMock.aMultipart;
import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.binaryEqualTo;
import static com.github.tomakehurst.wiremock.client.WireMock.equalTo;
import static com.github.tomakehurst.wiremock.client.WireMock.equalToJson;
import static com.github.tomakehurst.wiremock.client.WireMock.post;
import static com.github.tomakehurst.wiremock.client.WireMock.postRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.springframework.web.client.RestClient;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.port.ClipDelivery;
import com.github.tomakehurst.wiremock.junit5.WireMockExtension;

import java.util.Optional;

class TelegramBotClientTest {

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance().build();

    private static final String SEND = "/bottest-token/sendMessage";
    private static final String EDIT = "/bottest-token/editMessageText";
    private static final String ANSWER = "/bottest-token/answerCallbackQuery";

    private TelegramBotClient client;

    @BeforeEach
    void setUp() {
        TelegramConfig properties =
                new TelegramConfig("test-token", "12345", wireMock.baseUrl(), "{roomName}", "{roomName}");
        client = new TelegramBotClient(properties, RestClient.builder());
    }

    private static String okWithMessageId(long messageId) {
        return "{\"ok\":true,\"result\":{\"message_id\":" + messageId + "}}";
    }

    private static final String ANIMATION = "/bottest-token/sendAnimation";
    private static final byte[] CLIP_BYTES = "fake-mp4-bytes".getBytes(java.nio.charset.StandardCharsets.UTF_8);

    private static ClipContent clip() {
        return new ClipContent("FE-1.mp4", "video/mp4", CLIP_BYTES.length,
                () -> new java.io.ByteArrayInputStream(CLIP_BYTES));
    }

    @Test
    void sendsTheClipAsAnAnimationReplyingToTheAlertMessage() {
        wireMock.stubFor(post(urlEqualTo(ANIMATION)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"ok\":true,\"result\":{\"message_id\":601,"
                        + "\"animation\":{\"file_id\":\"file-abc\"}}}")));

        Optional<ClipDelivery> delivery = client.sendClipReply(501L, clip());

        assertThat(delivery).isPresent();
        assertThat(delivery.orElseThrow().fileId()).isEqualTo("file-abc");
        assertThat(delivery.orElseThrow().messageId()).isEqualTo(601L);

        wireMock.verify(postRequestedFor(urlEqualTo(ANIMATION))
                .withRequestBodyPart(aMultipart().withName("chat_id")
                        .withBody(equalTo("12345")).build())
                .withRequestBodyPart(aMultipart().withName("reply_to_message_id")
                        .withBody(equalTo("501")).build())
                .withRequestBodyPart(aMultipart().withName("allow_sending_without_reply")
                        .withBody(equalTo("true")).build())
                .withRequestBodyPart(aMultipart().withName("animation")
                        .withBody(binaryEqualTo(CLIP_BYTES)).build()));
    }

    @Test
    void fallsBackToTheDocumentFileIdWhenTelegramDoesNotTreatItAsAnAnimation() {
        wireMock.stubFor(post(urlEqualTo(ANIMATION)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"ok\":true,\"result\":{\"message_id\":602,"
                        + "\"document\":{\"file_id\":\"doc-xyz\"}}}")));

        assertThat(client.sendClipReply(501L, clip()).orElseThrow().fileId()).isEqualTo("doc-xyz");
    }

    @Test
    void clipSendIsBestEffortAndReturnsEmptyOnFailure() {
        wireMock.stubFor(post(urlEqualTo(ANIMATION)).willReturn(aResponse().withStatus(500)));

        assertThat(client.sendClipReply(501L, clip())).isEmpty();
    }

    @Test
    void clipSendReturnsEmptyWhenTelegramReportsNotOk() {
        wireMock.stubFor(post(urlEqualTo(ANIMATION)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"ok\":false}")));

        assertThat(client.sendClipReply(501L, clip())).isEmpty();
    }

    @Test
    void sendsAlertWithChatIdHtmlParseModeAndAcknowledgeButton() {
        wireMock.stubFor(post(urlEqualTo(SEND)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody(okWithMessageId(42))));

        long messageId = client.sendAlert(new EventId("FE-1"), "Fall in Block A - Room 12");

        assertThat(messageId).isEqualTo(42L);
        wireMock.verify(postRequestedFor(urlEqualTo(SEND))
                .withRequestBody(equalToJson("""
                        {
                          "chat_id": "12345",
                          "text": "Fall in Block A - Room 12",
                          "parse_mode": "HTML",
                          "reply_markup": {
                            "inline_keyboard": [[{"text": "✅ Acknowledge", "callback_data": "ack:FE-1"}]]
                          }
                        }""")));
    }

    @Test
    void sendsTriageQuestionWithBothOutcomeButtonsOnOneRow() {
        wireMock.stubFor(post(urlEqualTo(SEND)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody(okWithMessageId(43))));

        long messageId = client.sendTriageQuestion(new EventId("FE-1"), "Was this a genuine fall?");

        assertThat(messageId).isEqualTo(43L);
        wireMock.verify(postRequestedFor(urlEqualTo(SEND))
                .withRequestBody(equalToJson("""
                        {
                          "chat_id": "12345",
                          "text": "Was this a genuine fall?",
                          "parse_mode": "HTML",
                          "reply_markup": {
                            "inline_keyboard": [[
                              {"text": "👍 Genuine", "callback_data": "triage:FE-1:genuine"},
                              {"text": "🚫 False alarm", "callback_data": "triage:FE-1:false_alarm"}
                            ]]
                          }
                        }""")));
    }

    @Test
    void closeMessageSendsAnEmptyKeyboardToClearTheButtons() {
        wireMock.stubFor(post(urlEqualTo(EDIT)).willReturn(aResponse().withStatus(200)));

        client.closeMessage(42L, "✅ HANDLED");

        wireMock.verify(postRequestedFor(urlEqualTo(EDIT))
                .withRequestBody(equalToJson("""
                        {
                          "chat_id": "12345",
                          "message_id": 42,
                          "text": "✅ HANDLED",
                          "parse_mode": "HTML",
                          "reply_markup": {"inline_keyboard": []}
                        }""")));
    }

    @Test
    void closeMessageToleratesTelegramRejectingAnUnchangedEdit() {
        wireMock.stubFor(post(urlEqualTo(EDIT)).willReturn(aResponse()
                .withStatus(400)
                .withHeader("Content-Type", "application/json")
                .withBody("{\"ok\":false,\"error_code\":400,"
                        + "\"description\":\"Bad Request: message is not modified\"}")));

        assertThatCode(() -> client.closeMessage(42L, "✅ HANDLED")).doesNotThrowAnyException();
    }

    @Test
    void closeMessageIsBestEffortAndSwallowsTransportFailures() {
        wireMock.stubFor(post(urlEqualTo(EDIT)).willReturn(aResponse().withStatus(500)));

        assertThatCode(() -> client.closeMessage(42L, "✅ HANDLED")).doesNotThrowAnyException();
    }

    @Test
    void answerCallbackPostsTheQueryId() {
        wireMock.stubFor(post(urlEqualTo(ANSWER)).willReturn(aResponse().withStatus(200)));

        client.answerCallback("cbq-1", "Acknowledged");

        wireMock.verify(postRequestedFor(urlEqualTo(ANSWER))
                .withRequestBody(equalToJson(
                        "{\"callback_query_id\":\"cbq-1\",\"text\":\"Acknowledged\"}")));
    }
}
