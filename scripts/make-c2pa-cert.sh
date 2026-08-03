#!/usr/bin/env bash
# make-c2pa-cert.sh — generate a PER-INSTALLATION C2PA signing chain.
#
# CrispEmbed ships no signing key, on purpose. A private key published in an MIT
# repository would let anyone mint a manifest naming CrispEmbed as the software
# agent for an image it never touched, and re-sign after altering the pixels —
# destroying both jobs a C2PA signature exists to do while looking like it does
# them. This script gives you signing without that: the key is generated here,
# on this machine, and never leaves it.
#
#   ./scripts/make-c2pa-cert.sh [outdir]        # default: ~/.config/crispembed/c2pa
#   export CRISPEMBED_C2PA_CERT=<outdir>/cert.pem
#   export CRISPEMBED_C2PA_KEY=<outdir>/key.pem
#
# WHAT THIS DOES AND DOES NOT GIVE YOU. The chain is locally rooted, so it is
# not in the C2PA trust list and verifiers will show "unverified signer" — the
# manifest attests WHAT WAS DONE, not WHO DID IT. That is the same trust level
# a bundled certificate would give, minus the shared secret. For attributable
# provenance you need a certificate from a CA on the C2PA trust list.
#
# Two details that are not optional, both learned by watching c2pa-rs refuse:
#   * a SELF-SIGNED certificate is rejected outright — hence a leaf + CA
#   * the key must be PKCS#8 ("BEGIN PRIVATE KEY"), not SEC1 ("BEGIN EC
#     PRIVATE KEY"), or you get an opaque ASN.1 error
set -euo pipefail

OUT="${1:-$HOME/.config/crispembed/c2pa}"
mkdir -p "$OUT"
cd "$OUT"

if ! command -v openssl >/dev/null 2>&1; then
    echo "error: openssl not found" >&2
    exit 1
fi

if [ -f cert.pem ] && [ -f key.pem ]; then
    echo "Chain already present in $OUT — delete cert.pem/key.pem to regenerate."
    exit 0
fi

HOSTTAG="$(hostname -s 2>/dev/null || echo local)"

# Local CA.
openssl ecparam -name prime256v1 -genkey -noout -out ca.key
openssl req -new -x509 -key ca.key -out ca.pem -days 3650 \
    -subj "/CN=CrispEmbed Local CA (${HOSTTAG})" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign"

# Leaf. The emailProtection EKU is what c2pa-rs expects for a document signer.
openssl ecparam -name prime256v1 -genkey -noout -out leaf.key
openssl req -new -key leaf.key -out leaf.csr -subj "/CN=CrispEmbed (${HOSTTAG}, unverified)"
cat > leaf.ext <<'EXT'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature
extendedKeyUsage=critical,emailProtection
EXT
openssl x509 -req -in leaf.csr -CA ca.pem -CAkey ca.key -CAcreateserial \
    -out leaf.pem -days 3650 -extfile leaf.ext

openssl pkcs8 -topk8 -nocrypt -in leaf.key -out key.pem
cat leaf.pem ca.pem > cert.pem
chmod 600 key.pem leaf.key ca.key
rm -f leaf.csr leaf.ext

cat <<EOF

Wrote a local signing chain to $OUT

  export CRISPEMBED_C2PA_CERT="$OUT/cert.pem"
  export CRISPEMBED_C2PA_KEY="$OUT/key.pem"

Images will now carry a signed C2PA manifest. Verifiers will report an
unverified signer: this chain is locally rooted, so it proves what was done,
not who did it. Keep key.pem private — anyone holding it can sign as this
installation.
EOF
