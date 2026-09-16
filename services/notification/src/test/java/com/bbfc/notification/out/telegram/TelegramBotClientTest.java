package com.bbfc.notification.out.telegram;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.equalToJson;
import static com.github.tomakehurst.wiremock.client.WireMock.post;
import static com.github.tomakehurst.wiremock.client.WireMock.postRequestedFor;
import static com.github.tomakehurst.wiremock.client.WireMock.urlEqualTo;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.springframework.web.client.RestClient;

import com.bbfc.notification.config.TelegramConfig;
import com.github.tomakehurst.wiremock.junit5.WireMockExtension;

class TelegramBotClientTest {

    @RegisterExtension
    static WireMockExtension wireMock = WireMockExtension.newInstance().build();

    @Test
    void sendsMessageWithChatIdAndHtmlParseMode() {
        wireMock.stubFor(post(urlEqualTo("/bottest-token/sendMessage"))
                .willReturn(aResponse().withStatus(200)));

        TelegramConfig properties =
                new TelegramConfig("test-token", "12345", wireMock.baseUrl(), "{roomName}");
        TelegramBotClient client = new TelegramBotClient(properties, RestClient.builder());

        client.sendAlert("Fall in Block A - Room 12");

        wireMock.verify(postRequestedFor(urlEqualTo("/bottest-token/sendMessage"))
                .withRequestBody(equalToJson(
                        "{\"chat_id\":\"12345\",\"text\":\"Fall in Block A - Room 12\",\"parse_mode\":\"HTML\"}")));
    }
}
