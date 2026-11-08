package com.bbfc.notification;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;

/** Enforces the hexagonal rule in .gemini/styleguide.md: the core talks to adapters only through its ports. */
class ArchitectureTest {

    private static final Path CORE = Path.of("src/main/java/com/bbfc/notification/core");
    private static final Pattern OUTWARD_IMPORT = Pattern.compile(
            "^import (static )?com\\.bbfc\\.notification\\.(in|out|config)\\.\\S+;", Pattern.MULTILINE);

    @Test
    void coreImportsNothingFromAdaptersOrConfig() throws IOException {
        List<String> breaches;
        try (Stream<Path> files = Files.walk(CORE)) {
            breaches = files
                    .filter(file -> file.toString().endsWith(".java"))
                    .flatMap(file -> OUTWARD_IMPORT.matcher(read(file)).results()
                            .map(match -> CORE.relativize(file) + ": " + match.group()))
                    .toList();
        }

        assertThat(breaches)
                .as("core/ must not import in/, out/ or config/; hand it a domain type instead, as PolicyConfig does")
                .isEmpty();
    }

    @Test
    void theCoreSourcesAreWhereThisTestLooks() {
        assertThat(CORE).isDirectory();
    }

    private static String read(Path file) {
        try {
            return Files.readString(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
