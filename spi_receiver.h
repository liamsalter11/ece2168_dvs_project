/**
 * spi_receiver.h — TCP receiver emulating SPI bus reception of DVS events
 */

#ifndef SPI_RECEIVER_H
#define SPI_RECEIVER_H

#include "protocol.h"
#include <vector>
#include <cstddef>
#include <cstdint>

/**
 * Holds a deserialized frame of DVS events.
 */
struct DVSFrame {
    uint32_t frame_id;
    uint32_t event_count;
    std::vector<dvs_event_t> events;
};

/**
 * Emulates an SPI receiver by listening on a TCP socket and
 * deserializing DVS event packets from the Python sender.
 */
class SPIReceiver {
public:
    SPIReceiver(int port);
    ~SPIReceiver();

    /** Start listening and block until a client connects. */
    bool listen_and_accept();

    /** Read one complete frame packet. Returns false on disconnect/error. */
    bool receive_frame(DVSFrame& out_frame);

    /** Close all sockets. */
    void shutdown();

private:
    int port_;
    int listen_fd_;
    int client_fd_;

    /** Read exactly n bytes from client_fd_ into buf. */
    bool read_exact(uint8_t* buf, size_t n);
};

#endif /* SPI_RECEIVER_H */
