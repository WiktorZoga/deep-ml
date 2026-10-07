def compare_ipc_methods(methods: list, message_size_bytes: int, num_messages: int) -> list:
    """
    Compare IPC methods and rank them by estimated total time.
    
    Args:
        methods: List of dicts with keys 'name', 'latency_us', 'bandwidth_mbps', 'setup_cost_us'
        message_size_bytes: Size of each message in bytes
        num_messages: Number of messages to transfer
    
    Returns:
        List of dicts with 'name', 'total_time_us', 'throughput_msgs_per_sec',
        sorted by total_time_us ascending.
    """

    def collect_stats(method):
        transfer_time = message_size_bytes / method['bandwidth_mbps']
        per_msg =  method['latency_us'] + transfer_time
        total = method['setup_cost_us'] + num_messages * per_msg
        throughput = num_messages * 1e6 / total

        return {
            'name': method['name'],
            'total_time_us': round(total, 2),
            'throughput_msgs_per_sec': round(throughput, 2),
        }
    
    return sorted([collect_stats(method) for method in methods], key=lambda x: x["total_time_us"])

