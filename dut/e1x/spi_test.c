/* Minimal SPI slave loopback test for E1x EVK.
 *
 * Receives 4 bytes over SPI_2 (Arduino UNO header, PINMUX_2) and prints
 * them as hex on the STDIO UART.  Run with the companion spi_test_sender.py
 * on the Windows side to verify SPI slave receive and UART output work.
 */
#include <eff.h>
#include <eff/drivers/spi.h>
#include <eff/drivers/pinmux.h>
#include <eff/drivers/uart.h>
#include <eff/time.h>
#include <stdio.h>
#include <stdint.h>

int main(void)
{
    eff_uart_cfg_t uart_cfg = EFF_UART_DEFAULTS;
    uart_cfg.baud = 108000;
    if (eff_uart_init(STDIO_UART, uart_cfg)) {
        return -1;
    }
    sleep_ms(10);

    eff_pinmux_set(PINMUX_2, PINMUX_SPI);

    eff_spi_slave_cfg_t spi_cfg = EFF_SPI_SLAVE_DEFAULTS;
    spi_cfg.proto     = SPI_SLAVE_DATA_ONLY;
    spi_cfg.xfer_mode = SPI_XFER_READ_ONLY;
    spi_cfg.bus_size  = SPI_BUS_SINGLE;
    spi_cfg.mode      = SPI_MODE_0;
    eff_spi_slave_init(SPI_2, &spi_cfg);

    eff_uart_puts(STDIO_UART, "SPI test ready\r\n");

    static uint8_t buf[4];
    static char line[32];
    for (;;) {
        int8_t rc = eff_spi_slave_xfer(SPI_2, NULL, 0, buf, sizeof(buf));
        if (rc == -2) {
            eff_uart_puts(STDIO_UART, "overrun\r\n");
            continue;
        }
        sprintf(line, "rx: %02X %02X %02X %02X\r\n",
                buf[0], buf[1], buf[2], buf[3]);
        eff_uart_puts(STDIO_UART, line);
    }

    return 0;
}
