import os
import pcap
import dpkt
from math import log10
import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append("cfg")
import cfg

# Configurations and Constants
OUTPUT_DIR = '../data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BW = cfg.EXTRACTOR_CONFIG['bandwidth']
SAMPLING_DECIMALS = int(log10(cfg.EXTRACTOR_CONFIG['SAMPLE']))

# Number of subcarriers calculation
NUM_SUBCARRIERS = int(BW * 3.2)

# Sniffing duration in seconds
SNIFF_DURATION = 40

def round_down(num, precision):
    """ Round down the number to specified decimal places """
    factor = 10 ** precision
    return float(int(num * factor) / factor)

def capture_csi(interface):
    print(f'Initializing CSI capture on {interface}...')
    start_time = datetime.now()
    packet_sniffer = pcap.pcap(name=interface, promisc=True, immediate=True, timeout_ms=50)
    packet_sniffer.setfilter('udp port 5500')

    # DataFrame Headers
    headers = ['mac_address', 'timestamp'] + ['sc_' + str(i) for i in range(NUM_SUBCARRIERS)]

    csi_data_by_mac = {}
    last_timestamp = 0.0

    for timestamp, packet in packet_sniffer:
        if int(timestamp) == int(last_timestamp):
            current_ts = round_down(timestamp, SAMPLING_DECIMALS)
            last_ts = round_down(last_timestamp, SAMPLING_DECIMALS)

            if current_ts == last_ts:
                last_timestamp = timestamp
                continue

        eth_frame = dpkt.ethernet.Ethernet(packet)
        ip_packet = eth_frame.data
        udp_segment = ip_packet.data

        mac_addr = udp_segment.data[4:10].hex()

        if mac_addr not in csi_data_by_mac:
            csi_data_by_mac[mac_addr] = pd.DataFrame(columns=headers)

        csi_raw = udp_segment.data[18:]

        bandwidth = ip_packet.__hdr__[2][2]
        num_sub = int(bandwidth * 3.2)

        csi_arr = np.frombuffer(csi_raw, dtype=np.int16, count=num_sub * 2).reshape((1, num_sub * 2))
        csi_complex = np.fft.fftshift(csi_arr[:1, ::2] + 1j * csi_arr[:1, 1::2], axes=(1,))

        csi_entry = pd.DataFrame(csi_complex)
        csi_entry.insert(0, 'mac_address', mac_addr)
        csi_entry.insert(1, 'timestamp', timestamp)

        subcarrier_cols = {i: 'sc_' + str(i) for i in range(num_sub)}
        csi_entry.rename(columns=subcarrier_cols, inplace=True)

        try:
            csi_data_by_mac[mac_addr] = pd.concat([csi_data_by_mac[mac_addr], csi_entry], ignore_index=True)
        except Exception as error:
            print('Concatenation Error:', error)

        last_timestamp = timestamp

        if (datetime.now() - start_time).seconds >= SNIFF_DURATION:
            print("Ending CSI Data Collection...")

            for mac, df in csi_data_by_mac.items():
                file_number = input("Enter a number to append to the file name: ")
                file_name = f'csi_walk_{file_number}.csv'
                save_path = os.path.join(OUTPUT_DIR, file_name)
                df.to_csv(save_path, index=False)
                print(f"File saved: {save_path}")
            break

if __name__ == '__main__':
    
    capture_csi('wlan0')
