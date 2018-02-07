python2 plot_props.py --log_dir logs/test_multi_boost
python2 plot_props.py --log_dir logs/test_multi_dampen

cp logs/test_multi_dampen/G_fix199000.png temp/dampen_sample.png
cp logs/test_multi_dampen/source_and_target_barplot.png temp/dampen_plots.png
cp logs/test_multi_boost/G_fix199000.png temp/boost_sample.png
cp logs/test_multi_boost/source_and_target_barplot.png temp/boost_plots.png

# echo $PWD -- {} | mutt momod@utexas.edu -s "Plots and samples" -a temp/*.png
echo $PWD -- {} | mutt guywcole@utexas.edu -s "Plots and samples" -a temp/*.png

