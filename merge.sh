i=0;
out='';
for file in $@; 
do
    tmp="$(tail -n +3 $file | ghead -n -2)";
    # if ((i > 0)); then
    #     tmp=`cut -d \& -f2,3 < $tmp`;
    # fi 
    # if ((i != 2)); then
    #     tmp=`sed -E 's|(\\){2}|\&|g' < $tmp`;
    # fi
    # i=$i+1;
    # paste <(echo $out) <(echo $tmp) | column -s $'\t' -t
    paste <(echo "$tmp");
done
# echo $out;