function selection_sort(arr) {
    for(let i = 0; i < arr.length - 1; i++) {
        let min_idx = i;
        for(let j = i + 1; j < arr.length; j++) {
            if(arr[j] < arr[i]) {
                min_idx = j;
            }
        }
        if(min_idx != i) {
            let tmp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = tmp;
        }
    }

    return arr;
}

let arr1 = [1, 3, 4, 2, 5, 5, 4, 10]
let arr2 = selection_sort(arr1)

arr2.forEach((element)=> {
    console.log(element);
});
