mkdir -p rendered_graphs

for file in *.dot; do
    output="rendered_graphs/${file%.dot}.pdf"
    dot -Gmode=KK -Tpdf "$file" -o "$output"
    echo "Rendered $file to $output"
done