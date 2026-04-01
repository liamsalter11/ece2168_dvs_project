/**
 * spi_receiver.cpp — TCP-based SPI bus emulator implementation
 */

#include "spi_receiver.h"

#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

SPIReceiver::SPIReceiver(int port)
    : port_(port), listen_fd_(-1), client_fd_(-1)
{
}

SPIReceiver::~SPIReceiver()
{
    shutdown();
}

bool SPIReceiver::listen_and_accept()
{
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        perror("[receiver] socket()");
        return false;
    }

    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (bind(listen_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("[receiver] bind()");
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    if (listen(listen_fd_, 1) < 0) {
        perror("[receiver] listen()");
        close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    printf("[receiver] Listening on port %d ...\n", port_);

    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    client_fd_ = accept(listen_fd_, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd_ < 0) {
        perror("[receiver] accept()");
        return false;
    }

    /* Disable Nagle for low-latency */
    int flag = 1;
    setsockopt(client_fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    printf("[receiver] Client connected from %s:%d\n",
           inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));

    return true;
}

bool SPIReceiver::read_exact(uint8_t* buf, size_t n)
{
    size_t total = 0;
    while (total < n) {
        ssize_t r = read(client_fd_, buf + total, n - total);
        if (r <= 0) {
            if (r == 0) {
                printf("[receiver] Client disconnected.\n");
            } else {
                perror("[receiver] read()");
            }
            return false;
        }
        total += (size_t)r;
    }
    return true;
}

bool SPIReceiver::receive_frame(DVSFrame& out_frame)
{
    /* Read 4-byte length prefix */
    uint8_t len_buf[4];
    if (!read_exact(len_buf, 4))
        return false;

    uint32_t packet_len;
    memcpy(&packet_len, len_buf, 4);

    if (packet_len < DVS_HEADER_SIZE || packet_len > (DVS_HEADER_SIZE + DVS_MAX_EVENTS * DVS_EVENT_SIZE)) {
        printf("[receiver] Invalid packet length: %u\n", packet_len);
        return false;
    }

    /* Read the full packet */
    std::vector<uint8_t> packet(packet_len);
    if (!read_exact(packet.data(), packet_len))
        return false;

    /* Parse header */
    dvs_packet_header_t header;
    memcpy(&header, packet.data(), sizeof(header));

    if (header.magic[0] != DVS_MAGIC_0 || header.magic[1] != DVS_MAGIC_1) {
        printf("[receiver] Bad magic: 0x%02X 0x%02X\n", header.magic[0], header.magic[1]);
        return false;
    }

    /* Validate event count */
    size_t expected_size = DVS_HEADER_SIZE + header.event_count * DVS_EVENT_SIZE;
    if (packet_len != expected_size) {
        printf("[receiver] Size mismatch: got %u, expected %zu (events=%u)\n",
               packet_len, expected_size, header.event_count);
        return false;
    }

    /* Parse events */
    out_frame.frame_id = header.frame_id;
    out_frame.event_count = header.event_count;
    out_frame.events.resize(header.event_count);

    const uint8_t* ptr = packet.data() + DVS_HEADER_SIZE;
    for (uint32_t i = 0; i < header.event_count; i++) {
        memcpy(&out_frame.events[i], ptr, DVS_EVENT_SIZE);
        ptr += DVS_EVENT_SIZE;
    }

    return true;
}

void SPIReceiver::shutdown()
{
    if (client_fd_ >= 0) {
        close(client_fd_);
        client_fd_ = -1;
    }
    if (listen_fd_ >= 0) {
        close(listen_fd_);
        listen_fd_ = -1;
    }
}
