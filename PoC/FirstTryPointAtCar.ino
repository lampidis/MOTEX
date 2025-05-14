#include <Adafruit_NeoPixel.h>

#define LED_PIN     6
#define NUM_LEDS    200
#define BRIGHTNESS  150

#define LED_START   65  // First usable LED
#define LED_END     135 // Last usable LED

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB);

void setup() {
    Serial.begin(9600);
    strip.begin();
    strip.setBrightness(BRIGHTNESS);
    clearLEDs();
}

void loop() {
    if (Serial.available()) {
        int ledIndex = Serial.parseInt();  // Read LED index from Python

        if (ledIndex >= LED_START && ledIndex <= LED_END) {
            clearLEDs();
            strip.setPixelColor(ledIndex, strip.Color(255, 0, 0));  // Red LED
            strip.show();
        }
    }
}

// Function to turn off all LEDs
void clearLEDs() {
    strip.clear();
    strip.show();
}
