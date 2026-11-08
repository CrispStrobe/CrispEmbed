| recipe | MiB | final cos min (case) | per case | top1 agree min | free-run IDs exact | text = F16 (boxes stripped) | max box delta | wall s |
|---|---|---|---|---|---|---|---|---|
| f16 | - | - |  | - | 4/4 | 4/4 | 0 | 1287 |
| f16cuda | - | 0.9789 (german) | paragraphs:0.9989 table:0.9941 receipt:0.9989 german:0.9789 | 1.0000 | 4/4 | 4/4 | 0 | 134 |
| q8 | 3397.4 | 0.9700 (paragraphs) | paragraphs:0.9700 table:0.9867 receipt:0.9882 german:0.9863 | 0.9902 | 1/4 | 4/4 | 3 | 120 |
| q8at16 | 3467.7 | 0.9726 (receipt) | paragraphs:0.9954 table:0.9849 receipt:0.9726 german:0.9802 | 0.9818 | 2/4 | 4/4 | 3 | 117 |
| q8hd16 | 3693.3 | 0.9737 (receipt) | paragraphs:0.9882 table:0.9833 receipt:0.9737 german:0.9881 | 0.9928 | 2/4 | 4/4 | 3 | 119 |
| q8sa16 | 3535.4 | 0.9791 (german) | paragraphs:0.9891 table:0.9856 receipt:0.9945 german:0.9791 | 0.9928 | 1/4 | 4/4 | 3 | 118 |
| q8sh16 | 3465.1 | 0.9762 (receipt) | paragraphs:0.9930 table:0.9774 receipt:0.9762 german:0.9949 | 0.9818 | 0/4 | 4/4 | 3 | 121 |
