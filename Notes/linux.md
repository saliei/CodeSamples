* write iso to usb to live boot:
    - first zero out the partition table, then create a Linux filesystem on the usb

`> dd if=/dev/zero of=/dev/sdb count=1 bs=512`

```bash
fdisk /dev/sdb <<EOF
n
p
1


t
b
w
EOF
```

`> mkdosfs -F32 /dev/sdb1`

---
