#! /bin/bash
### XRD txt file conversion and cleaning script

# for each file in the list,

# replace all the spaces with underscores
for file in ./txt_files/txt/*; do
    mv "$file" "${file// /_}"
done

# replace TXT with txt
for file in ./txt_files/txt/*; do
    mv "$file" "${file//TXT/txt}"
done

mkdir -p ./txt_files/utf8

# # for each file in the list,
for file in ./txt_files/txt/*; do
    # get file encoding 
    encoding=$(file -i $file | cut -d "=" -f 2)
    echo $encoding

    # convert the file to utf-8
    iconv -f $encoding -t utf-8 $file > ./txt_files/utf8/$(basename $file)
done