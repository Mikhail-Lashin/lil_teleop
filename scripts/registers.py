import Register.RegisterKey.ftp_registers_keys as ftp_keys

print("\n>>> All registers in Register.RegisterKey.ftp_registers_keys: \n")

all_attrs = [attr for attr in dir(ftp_keys) if not attr.startswith("__")]
for attr in sorted(all_attrs):
    val = getattr(ftp_keys, attr)
    if attr not in ["REGISTER_MAP", "RegisterName", "ALL_REGISTER_NAMES"]:
        print(f">>> {attr:<35} = {val}")