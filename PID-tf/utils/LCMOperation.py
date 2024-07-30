from datetime import datetime
import socket
import constants

_channels_count = constants.LCM_CHANNELS_COUNT
_available_channels_count = constants.LCM_AVAILABLE_CHANNELS_COUNT


def create_client():
    return socket.create_connection(('192.168.123.1', 8000), 500)


def operate_a103(client: socket.socket):
    s_buffer = b'\xaa\x55\x00\xff\x03\xa1\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    client.send(s_buffer)
    r_buffers = bytearray()
    bytes_read, total_len = 0, 12 + 52 + 52 * _channels_count + 4
    while bytes_read < total_len:
        r_buffer = client.recv(total_len - bytes_read)
        if r_buffer == b'':
            return
        r_buffers.extend(r_buffer)
        bytes_read += len(r_buffer)
    absolute_time = datetime.now()
    total_time = None
    current_step_time = None
    work_state = None
    step_index = None
    step_type = None
    error_code = None
    voltages = []
    currents = []
    line_voltages = []
    capacities = []
    energies = []
    powers = []
    temperatures = []
    battery_temperatures = []
    fan_state = None
    contact_impedances = []
    for i in range(_available_channels_count):
        start_index = 12 + 52 + 52 * i
        total_time = total_time or int.from_bytes(r_buffers[start_index: start_index + 4], 'little')
        current_step_time = current_step_time or int.from_bytes(r_buffers[start_index + 4: start_index + 8], 'little')
        work_state = work_state or r_buffers[start_index + 8]
        step_index = step_index or r_buffers[start_index + 9]
        step_type = step_type or r_buffers[start_index + 10]
        error_code = error_code or r_buffers[start_index + 23]
        voltages.append(int.from_bytes(r_buffers[start_index + 24: start_index + 28], 'little') / 100)
        currents.append(int.from_bytes(r_buffers[start_index + 28: start_index + 32], 'little') / 100)
        line_voltages.append(int.from_bytes(r_buffers[start_index + 32: start_index + 36], 'little') / 100)
        capacities.append(int.from_bytes(r_buffers[start_index + 36: start_index + 40], 'little') / 100)
        energies.append(int.from_bytes(r_buffers[start_index + 40: start_index + 44], 'little') / 100)
        temperatures.append(int.from_bytes(r_buffers[start_index + 44: start_index + 46], 'little') / 100)
        battery_temperatures.append(int.from_bytes(r_buffers[start_index + 46: start_index + 48], 'little') / 100)
        fan_state = fan_state or r_buffers[start_index + 48]
        contact_impedances.append(int.from_bytes(r_buffers[start_index + 50: start_index + 52], 'little') / 100)
    return absolute_time \
        , total_time \
        , current_step_time \
        , step_index \
        , step_type \
        , sum(voltages) / _available_channels_count \
        , sum(currents) / _available_channels_count \
        , sum(line_voltages) / _available_channels_count \
        , sum(capacities) / _available_channels_count \
        , sum(energies) / _available_channels_count \
        , sum(powers) / _available_channels_count \
        , sum(temperatures) / _available_channels_count \
        , sum(battery_temperatures) / _available_channels_count \
        , 0 \
        , 30 \
        , sum(contact_impedances) / _available_channels_count

def operate_b103(client: socket.socket):
    s_buffer = b'\xaa\x55\x00\xff\x03\xb1\x00\x00\x0c\x00\x00\x00\x52\x01\x00\x00\x02\x00\x00\x00\x55\x00\x00\x00'
    client.send(s_buffer)
    r_buffers = bytearray()
    bytes_read, total_len = 0, 26
    while bytes_read < total_len:
        r_buffer = client.recv(total_len - bytes_read)
        if r_buffer == b'':
            return
        r_buffers.extend(r_buffer)
        bytes_read += len(r_buffer)
    start_index = 12 + 8
    return int.from_bytes(r_buffers[start_index: start_index + 2], 'little')

def operate_b101(client: socket.socket, pwm: int):
    s_buffers = bytearray(b'\xaa\x55\x00\xff\x01\xb1\x00\x00\x0e\x00\x00\x00')
    s_buffers.extend((8).to_bytes(4, 'little'))
    s_buffers.extend((2).to_bytes(4, 'little'))
    s_buffers.append(85)
    s_buffers.append(pwm)
    s_buffers.extend(sum(s_buffers[12:]).to_bytes(4, 'little'))
    client.send(s_buffers)
    client.recv(24)
