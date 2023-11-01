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

# Configuration and Constants
OUTPUT_DIR = '../data'  # Directory for saving output files
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create the directory if it doesn't exist

# Extract bandwidth and sampling rate from configuration file
BW = cfg.EXTRACTOR_CONFIG['bandwidth']
SAMPLING_DECIMALS = int(log10(cfg.EXTRACTOR_CONFIG['SAMPLE']))

# Calculate the number of subcarriers based on bandwidth
NUM_SUBCARRIERS = int(BW * 3.2)

# Sniffing duration for capturing packets (in seconds)
SNIFF_DURATION = 40

def round_down(num, precision):
    """ Round down a number to specified decimal places. """
    factor = 10 ** precision
    return float(int(num * factor) / factor)

def capture_csi(interface):
    """
    Capture Channel State Information (CSI) from network packets on a specified interface.

    Parameters:
    - interface: Network interface to use for capturing packets
    """
    print(f'Initializing CSI capture on {interface}...')
    start_time = datetime.now()

    # Set up packet sniffer on the specified network interface
    packet_sniffer = pcap.pcap(name=interface, promisc=True, immediate=True, timeout_ms=50)
    packet_sniffer.setfilter('udp port 5500')  # Filter for UDP packets on port 5500

    # Initialize DataFrame headers for storing CSI data
    headers = ['mac_address', 'timestamp'] + ['sc_' + str(i) for i in range(NUM_SUBCARRIERS)]
    csi_data_by_mac = {}  # Dictionary to store CSI data for each MAC address
    last_timestamp = 0.0  # Variable to track the last processed timestamp

    # Process captured packets
    for timestamp, packet in packet_sniffer:
        # Check for duplicate timestamp entries
        if int(timestamp) == int(last_timestamp):
            current_ts = round_down(timestamp, SAMPLING_DECIMALS)
            last_ts = round_down(last_timestamp, SAMPLING_DECIMALS)

            if current_ts == last_ts:
                last_timestamp = timestamp
                continue

        # Parse packet layers (Ethernet, IP, UDP)
        eth_frame = dpkt.ethernet.Ethernet(packet)
        ip_packet = eth_frame.data
        udp_segment = ip_packet.data

        # Extract MAC address from the packet
        mac_addr = udp_segment.data[4:10].hex()

        # Initialize a new DataFrame for a new MAC address
        if mac_addr not in csi_data_by_mac:
            csi_data_by_mac[mac_addr] = pd.DataFrame(columns=headers)

        # Extract raw CSI data from the packet
        csi_raw = udp_segment.data[18:]

        # Convert raw CSI data to complex numbers
        csi_arr = np.frombuffer(csi_raw, dtype=np.int16, count=NUM_SUBCARRIERS * 2).reshape((1, NUM_SUBCARRIERS * 2))
        csi_complex = np.fft.fftshift(csi_arr[:1, ::2] + 1j * csi_arr[:1, 1::2], axes=(1,))

        # Prepare DataFrame entry for the current packet's CSI data
        csi_entry = pd.DataFrame(csi_complex)
        csi_entry.insert(0, 'mac_address', mac_addr)
        csi_entry.insert(1, 'timestamp', timestamp)

        # Rename DataFrame columns to match subcarrier indices
        subcarrier_cols = {i: 'sc_' + str(i) for i in range(NUM_SUBCARRIERS)}
        csi_entry.rename(columns=subcarrier_cols, inplace=True)

        # Append the CSI entry to the DataFrame of the corresponding MAC address
        try:
            csi_data_by_mac[mac_addr] = pd.concat([csi_data_by_mac[mac_addr], csi_entry], ignore_index=True)
        except Exception as error:
            print('Concatenation Error:', error)

        # Update the last timestamp processed
        last_timestamp = timestamp

        # Check if the sniffing duration has been reached
        if (datetime.now() - start_time).seconds >= SNIFF_DURATION:
            print("Ending CSI Data Collection...")

            # Save CSI data for each MAC address to a CSV file
            for mac, df in csi_data_by_mac.items():
                file_number = input("Enter a number to append to the file name: ")
                file_name = f'csi_walk_{file_number}.csv'
                save_path = os.path.join(OUTPUT_DIR, file_name)
                df.to_csv(save_path, index=False)
                print(f"File saved: {save_path}")
            break

if __name__ == '__main__':
    # Start CSI data capturing on the specified network interface
    capture_csi('wlan0')
