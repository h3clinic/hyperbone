# Parent Supervision Debug

Checkpoint: outputs/models/hyperbone_anymate_static_v2.8_parent_probe/best_model.pt
Backbone: dgcnn
Threshold: 0.7

## Train
### direct
- GT joint count: 47.30
- Matched predicted count: 0.00
- Valid parent target count: 47.30
- Root target count: 1.15
- Non-root parent target count: 46.15
- Percent parent targets ROOT: 0.034
- Percent parent targets ignored: 0.630
- GT edges recoverable after matching: 0.00
- Parent target min/max: 0.00 / 128.00
- Raw parent cycles: 0.05
- Normalized parent cycles: 0.00
- Raw root count: 1.10
- Normalized root count: 1.15
- Edges preserved ratio: 0.999
- Samples requiring normalization: 0.05
- Normalization cycles detected/broken: 0.05 / 0.05
- Normalization invalid/self parents: 0.00 / 1.10
- Normalization no-root components / roots added: 0.00 / 0.00
- Normalization edges preserved/removed: 46.15 / 0.05
- Parent class distribution: {"0": 22, "1": 20, "10": 18, "11": 18, "12": 18, "128": 23, "13": 18, "14": 20, "15": 22, "16": 19, "17": 18, "18": 15, "19": 15, "2": 20, "20": 18, "21": 15, "22": 17, "23": 14, "24": 16, "25": 14, "26": 17, "27": 13, "28": 15, "29": 15, "3": 20, "30": 17, "31": 12, "32": 12, "33": 12, "34": 16, "35": 12, "36": 23, "37": 11, "38": 9, "39": 9, "4": 20, "40": 11, "41": 9, "42": 9, "43": 9, "44": 9, "45": 9, "46": 12, "47": 14, "48": 10, "49": 7, "5": 20, "50": 9, "51": 9, "52": 17, "53": 17, "54": 12, "55": 19, "56": 6, "57": 4, "58": 3, "59": 1, "6": 20, "60": 11, "61": 23, "62": 7, "63": 5, "65": 2, "67": 5, "69": 5, "7": 21, "8": 20, "9": 18}

### hungarian
- GT joint count: 47.30
- Matched predicted count: 47.30
- Valid parent target count: 47.30
- Root target count: 1.15
- Non-root parent target count: 46.15
- Percent parent targets ROOT: 0.034
- Percent parent targets ignored: 0.000
- GT edges recoverable after matching: 46.15
- Parent target min/max: 6.80 / 128.00
- Raw parent cycles: 0.00
- Normalized parent cycles: 0.00
- Raw root count: 0.00
- Normalized root count: 0.00
- Edges preserved ratio: 0.000
- Samples requiring normalization: 0.00
- Normalization cycles detected/broken: 0.00 / 0.00
- Normalization invalid/self parents: 0.00 / 0.00
- Normalization no-root components / roots added: 0.00 / 0.00
- Normalization edges preserved/removed: 0.00 / 0.00
- Parent class distribution: {"10": 5, "100": 10, "102": 20, "104": 20, "105": 2, "108": 1, "11": 1, "110": 18, "111": 15, "112": 13, "113": 8, "114": 16, "115": 9, "117": 12, "118": 16, "119": 18, "12": 11, "120": 15, "122": 8, "123": 9, "124": 8, "125": 1, "126": 1, "127": 15, "128": 23, "13": 8, "15": 1, "17": 18, "18": 14, "2": 17, "20": 17, "21": 1, "22": 1, "23": 1, "26": 9, "29": 11, "3": 1, "31": 8, "32": 12, "33": 2, "34": 17, "35": 10, "37": 8, "38": 14, "4": 7, "42": 12, "43": 9, "44": 2, "45": 1, "46": 18, "47": 8, "48": 18, "5": 17, "50": 20, "51": 15, "53": 15, "54": 11, "55": 7, "57": 1, "58": 1, "59": 14, "6": 7, "60": 9, "61": 20, "62": 14, "63": 9, "65": 4, "66": 3, "68": 1, "69": 20, "7": 4, "70": 20, "71": 19, "72": 12, "74": 3, "75": 8, "77": 19, "81": 8, "83": 1, "85": 15, "86": 9, "87": 19, "89": 13, "9": 16, "90": 9, "93": 20, "94": 10, "96": 9, "97": 6, "98": 18}

## Test
### direct
- GT joint count: 52.00
- Matched predicted count: 0.00
- Valid parent target count: 52.00
- Root target count: 1.25
- Non-root parent target count: 50.75
- Percent parent targets ROOT: 0.034
- Percent parent targets ignored: 0.594
- GT edges recoverable after matching: 0.00
- Parent target min/max: 0.00 / 128.00
- Raw parent cycles: 0.00
- Normalized parent cycles: 0.00
- Raw root count: 1.25
- Normalized root count: 1.25
- Edges preserved ratio: 1.000
- Samples requiring normalization: 0.00
- Normalization cycles detected/broken: 0.00 / 0.00
- Normalization invalid/self parents: 0.00 / 1.25
- Normalization no-root components / roots added: 0.00 / 0.00
- Normalization edges preserved/removed: 50.75 / 0.00
- Parent class distribution: {"0": 20, "1": 20, "10": 20, "11": 20, "12": 21, "128": 25, "13": 18, "14": 20, "15": 18, "16": 18, "17": 18, "18": 18, "19": 18, "2": 20, "20": 20, "21": 22, "22": 21, "23": 18, "24": 15, "25": 15, "26": 17, "27": 14, "28": 16, "29": 20, "3": 20, "30": 21, "31": 12, "32": 12, "33": 12, "34": 12, "35": 16, "36": 15, "37": 12, "38": 15, "39": 14, "4": 20, "40": 16, "41": 13, "42": 13, "43": 12, "44": 10, "45": 10, "46": 14, "47": 12, "48": 11, "49": 15, "5": 20, "50": 13, "51": 10, "52": 15, "53": 21, "54": 10, "55": 25, "56": 4, "57": 4, "58": 12, "59": 11, "6": 20, "60": 1, "61": 23, "62": 8, "64": 4, "65": 7, "68": 5, "7": 20, "71": 4, "72": 2, "78": 1, "8": 20, "81": 1, "9": 20}

### hungarian
- GT joint count: 52.00
- Matched predicted count: 52.00
- Valid parent target count: 52.00
- Root target count: 1.25
- Non-root parent target count: 50.75
- Percent parent targets ROOT: 0.034
- Percent parent targets ignored: 0.000
- GT edges recoverable after matching: 50.75
- Parent target min/max: 2.30 / 128.00
- Raw parent cycles: 0.00
- Normalized parent cycles: 0.00
- Raw root count: 0.00
- Normalized root count: 0.00
- Edges preserved ratio: 0.000
- Samples requiring normalization: 0.00
- Normalization cycles detected/broken: 0.00 / 0.00
- Normalization invalid/self parents: 0.00 / 0.00
- Normalization no-root components / roots added: 0.00 / 0.00
- Normalization edges preserved/removed: 0.00 / 0.00
- Parent class distribution: {"10": 6, "100": 11, "102": 19, "104": 20, "105": 3, "11": 3, "110": 18, "111": 17, "112": 14, "113": 10, "114": 18, "115": 13, "117": 15, "118": 17, "119": 20, "12": 11, "120": 16, "122": 10, "123": 10, "124": 10, "125": 2, "126": 1, "127": 16, "128": 25, "13": 8, "14": 1, "15": 1, "17": 20, "18": 14, "2": 18, "20": 18, "22": 3, "23": 2, "24": 1, "26": 10, "29": 13, "3": 2, "31": 11, "32": 13, "33": 4, "34": 14, "35": 13, "37": 9, "38": 16, "4": 8, "42": 13, "43": 11, "44": 2, "45": 1, "46": 19, "47": 8, "48": 18, "5": 19, "50": 20, "51": 18, "53": 17, "54": 12, "55": 8, "59": 16, "6": 7, "60": 10, "61": 20, "62": 12, "63": 10, "65": 3, "66": 3, "69": 20, "7": 8, "70": 20, "71": 18, "72": 13, "74": 4, "75": 10, "76": 1, "77": 20, "81": 10, "83": 1, "85": 17, "86": 11, "87": 19, "89": 14, "9": 20, "90": 11, "93": 20, "94": 12, "96": 11, "97": 9, "98": 20}
