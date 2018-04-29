package com.bellvenue.experimental.mnist;

import lombok.extern.slf4j.Slf4j;
import org.joda.time.DateTimeZone;
import org.springframework.boot.Banner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.core.env.Environment;

@SpringBootApplication
@ComponentScan("com.bellvenue.experimental")
@EnableAutoConfiguration
@Slf4j
public class Application {

    public static void main(String[] args) {
        DateTimeZone.setDefault(DateTimeZone.UTC);
        SpringApplication app = new SpringApplication(Application.class);
        app.setBannerMode(Banner.Mode.OFF);
        Environment env = app.run(args).getEnvironment();
    }
}
