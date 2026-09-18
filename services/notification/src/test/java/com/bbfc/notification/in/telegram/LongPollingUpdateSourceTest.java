package com.bbfc.notification.in.telegram;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.get;
import static com.github.tomakehurst.wiremock.client.WireMock.getRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.urlPathEqualTo;
import static com.github.tomakehurst.wiremock.client.WireMock.equalTo;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.web.client.RestClient;

import com.bbfc.notification.config.TelegramConfig;
import com.bbfc.notification.in.telegram.dto.TelegramUpdates.CallbackQuery;
import com.github.tomakehurst.wiremock.junit5.WireMockExtension;

@ExtendWith(MockitoExtension.class)
class LongPollingUpdateSourceTest {

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance().build();

    private static final String GET_UPDATES = "/bottest-token/getUpdates";

    @Mock
    private CallbackQueryHandler callbackQueryHandler;

    private LongPollingUpdateSource source;

    @BeforeEach
    void setUp() {
        wireMock.resetAll();
        TelegramConfig config =
                new TelegramConfig("test-token", "-100123", wireMock.baseUrl(), "{roomName}");
        source = new LongPollingUpdateSource(config, RestClient.builder(), callbackQueryHandler);
    }

    private static String updateWith(long updateId, String data) {
        return """
                {"ok":true,"result":[{
                  "update_id": %d,
                  "callback_query": {
                    "id": "cbq-1",
                    "from": {"id": 777, "username": "nurse_lim"},
                    "data": "%s",
                    "message": {"chat": {"id": -100123}}
                  }
                }]}""".formatted(updateId, data);
    }

    private void stubGetUpdates(String body) {
        wireMock.stubFor(get(urlPathEqualTo(GET_UPDATES)).willReturn(aResponse()
                .withStatus(200)
                .withHeader("Content-Type", "application/json")
                .withBody(body)));
    }

    @Test
    void handlesCallbacksAndAdvancesTheOffsetPastThem() {
        stubGetUpdates(updateWith(120, "ack:FE-1"));

        assertThat(source.pollOnce()).isTrue();

        verify(callbackQueryHandler).handle(any(CallbackQuery.class));
        assertThat(source.nextOffset()).isEqualTo(121L);
    }

    @Test
    void requestsOnlyCallbackQueriesAndSendsTheCursor() {
        stubGetUpdates("{\"ok\":true,\"result\":[]}");

        source.pollOnce();

        wireMock.verify(getRequestedFor(urlPathEqualTo(GET_UPDATES))
                .withQueryParam("offset", equalTo("0"))
                .withQueryParam("allowed_updates", equalTo("[\"callback_query\"]")));
    }

    @Test
    void anEmptyBatchLeavesTheOffsetAlone() {
        stubGetUpdates("{\"ok\":true,\"result\":[]}");

        assertThat(source.pollOnce()).isTrue();

        verify(callbackQueryHandler, never()).handle(any());
        assertThat(source.nextOffset()).isZero();
    }

    @Test
    void anHttpFailureBacksOffWithoutMovingTheOffset() {
        wireMock.stubFor(get(urlPathEqualTo(GET_UPDATES)).willReturn(aResponse().withStatus(500)));

        assertThat(source.pollOnce()).isFalse();

        assertThat(source.nextOffset()).isZero();
    }

    @Test
    void aConflictIsReportedButStillBacksOff() {
        wireMock.stubFor(get(urlPathEqualTo(GET_UPDATES)).willReturn(aResponse()
                .withStatus(409)
                .withBody("{\"ok\":false,\"error_code\":409,\"description\":\"Conflict\"}")));

        assertThat(source.pollOnce()).isFalse();
    }

    // Otherwise one bad update pins the cursor and the poller replays it forever.
    @Test
    void aFailingHandlerStillAdvancesTheOffset() {
        stubGetUpdates(updateWith(120, "ack:FE-1"));
        doThrow(new RuntimeException("boom")).when(callbackQueryHandler).handle(any());

        assertThat(source.pollOnce()).isTrue();

        assertThat(source.nextOffset()).isEqualTo(121L);
    }

    @Test
    void stopBeforeStartIsHarmless() {
        source.stop();
    }
}
