#!/usr/bin/env python3
import socket
import struct
import time


def main():
    host = "127.0.0.1"
    port = 5005
    fmt = "<IQffff"  # seq, t_ros_ns, x, y, z, confidence
    size = struct.calcsize(fmt)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"Listening on {host}:{port}, expecting {size} bytes per packet...")

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            if len(data) != size:
                print(f"Skipped packet from {addr} with unexpected size {len(data)}")
                continue

            seq, t_ros_ns, x, y, z, conf = struct.unpack(fmt, data)
            t_ros_s = t_ros_ns / 1e9
            print(
                f"seq={seq:10d} t_ros={t_ros_s:20.9f}s "
                f"x={x: .4f} y={y: .4f} z={z: .4f} conf={conf: .2f}"
            )
    except KeyboardInterrupt:
        print("\nInterrupted, exiting.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
