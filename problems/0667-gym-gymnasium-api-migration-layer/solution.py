def gym_api_convert(data, call_type: str, direction: str):
    """
    Convert between old Gym and new Gymnasium API output formats.

    Args:
        data: The output from reset() or step() in the source format.
        call_type: 'reset' or 'step'
        direction: 'old_to_new' or 'new_to_old'

    Returns:
        The converted output in the target API format.
    """
    if call_type == "reset":
        if direction == "old_to_new":
            # data to po prostu observation
            return (data, {})
        elif direction == "new_to_old":
            # data to (observation, info)
            return data[0]

    elif call_type == "step":
        if direction == "old_to_new":
            obs, reward, done, info = data
            info_out = info.copy()

            # Sprawdzamy czy powodem done był limit czasu
            was_truncated = bool(info_out.pop("TimeLimit.truncated", False))

            if was_truncated:
                truncated = True
                terminated = False
            else:
                truncated = False
                terminated = bool(done)

            return (obs, reward, terminated, truncated, info_out)

        elif direction == "new_to_old":
            obs, reward, terminated, truncated, info = data
            info_out = info.copy()

            done = bool(terminated or truncated)
            if truncated:
                info_out["TimeLimit.truncated"] = True
            else:
                info_out.pop("TimeLimit.truncated", None)

            return (obs, reward, done, info_out)

    raise ValueError(f"Nieobsługiwana kombinacja: {call_type}, {direction}")

