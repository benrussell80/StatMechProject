import pandas as pd
from bokeh.plotting import ColumnDataSource, figure
from bokeh.models import HoverTool, CategoricalColorMapper, Slider, Select, Label
from bokeh.models.widgets import Panel, Tabs, Div
from bokeh.models.glyphs import Text
from bokeh.io import curdoc, output_file, show
from bokeh.layouts import widgetbox, row, column
import numpy as np


csv_file = 'pulsar_stars.csv'

pulsar_stars_df = pd.read_csv(csv_file, header=0)

# fix entropy
def shannon_entropy(threshold, col_index):
    lt = pulsar_stars_df['target_class'].loc[pulsar_stars_df[pulsar_stars_df.columns[col_index]] < threshold]
    gte = pulsar_stars_df['target_class'].loc[pulsar_stars_df[pulsar_stars_df.columns[col_index]] >= threshold]
    
    probs_lesser = lt.value_counts().values/len(lt)
    probs_greater = gte.value_counts().values/len(gte)
    probs_all = pulsar_stars_df['target_class'].value_counts().values/len(pulsar_stars_df)

    parent_ent = sum([-p * np.log2(p) for p in probs_all])
    lesser_ent = sum([-p * np.log2(p) for p in probs_lesser])
    greater_ent = sum([-p * np.log2(p) for p in probs_greater])
    avg_weighted_ent = (lesser_ent * len(lt) + greater_ent * len(gte))/len(pulsar_stars_df)
    information_gain = parent_ent - avg_weighted_ent

    return parent_ent, lesser_ent, greater_ent, avg_weighted_ent, information_gain


# histogram data for both stars and non stars
h0_df = pulsar_stars_df.loc[pulsar_stars_df.target_class==0].apply(np.histogram, bins='sqrt')
h0_df = h0_df[h0_df.index[:-1]]
h0_quads = [{'top': x, 'left': y[:-1], 'right': y[1:]} for x, y in h0_df]

h1_df = pulsar_stars_df.loc[pulsar_stars_df.target_class==1].apply(np.histogram, bins='sqrt')
h1_df = h1_df[h1_df.index[:-1]]
h1_quads = [{'top': x, 'left': y[:-1], 'right': y[1:]} for x, y in h1_df]

tabs = []
sources = [ColumnDataSource(data={'x': [min(min(hists[0]['left']), min(hists[1]['left']))]*2, 'y': [0, max(max(hists[0]['top']), max(hists[1]['top']))]}) for i, hists in enumerate(zip(h0_quads, h1_quads))]

# first tab

slider1 = Slider(
    title='Threshold',
    start=min(pulsar_stars_df.iloc[:, 0]) - 0.1,
    end=max(pulsar_stars_df.iloc[:, 0]) + 0.1,
    step=(max(pulsar_stars_df.iloc[:, 0]) - min(pulsar_stars_df.iloc[:, 0]))/75,
    value=min(pulsar_stars_df.iloc[:, 0]) - 0.1
)

source1 = sources[0]

plot1 = figure()
plot1.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h0_quads[0]), color='red', legend='Not Star', fill_alpha=0.8)
plot1.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h1_quads[0]), color='green', legend='Star', fill_alpha=0.8)
plot1.legend.click_policy="hide"

igsource1 = ColumnDataSource(data={'x': np.linspace(slider1.start, slider1.end), 'y': np.array([shannon_entropy(x, 0)[4] for x in np.linspace(slider1.start, slider1.end)])})
igplot1 = figure(height=400, width=400, title='Information Gain vs Threshold')
igplot1.line(x='x', y='y', source=igsource1)
igmaxlabel1 = Label(
    x=igsource1.data['x'][igsource1.data['y'].argmax()],
    y=igsource1.data['y'].max(),
    text=f"Max Info Gain: ({igsource1.data['x'][igsource1.data['y'].argmax()]:.3f}, {igsource1.data['y'].max():.3f})",
    text_font_size='10pt')
igplot1.add_layout(igmaxlabel1)

div1 = Div(text=f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider1.value, 0)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider1.value, 0)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider1.value, 0)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider1.value, 0)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider1.value, 0)[4]:.3f}</li>
    </ul>""")

def callback1(attr, old, new):
    source1.data['x'] = [slider1.value, slider1.value]
    div1.text = f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider1.value, 0)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider1.value, 0)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider1.value, 0)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider1.value, 0)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider1.value, 0)[4]:.3f}</li>
    </ul>"""

slider1.on_change('value', callback1)

plot1.line(x='x', y='y', source=source1, line_width=3)
plot_and_slider1 = row(plot1, column(widgetbox(slider1), div1, igplot1))

tabs.append(Panel(child=plot_and_slider1, title=h1_df.index[0]))

# second tab

slider2 = Slider(
    title='Threshold',
    start=min(pulsar_stars_df.iloc[:, 1]) - 0.1,
    end=max(pulsar_stars_df.iloc[:, 1]) + 0.1,
    step=(max(pulsar_stars_df.iloc[:, 1]) - min(pulsar_stars_df.iloc[:, 1]))/75,
    value=min(pulsar_stars_df.iloc[:, 1]) - 0.1
)

source2 = sources[1]

plot2 = figure()
plot2.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h0_quads[1]), color='red', legend='Not Star', fill_alpha=0.8)
plot2.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h1_quads[1]), color='green', legend='Star', fill_alpha=0.8)
plot2.legend.click_policy="hide"

igsource2 = ColumnDataSource(data={'x': np.linspace(slider2.start, slider2.end), 'y': np.array([shannon_entropy(x, 1)[4] for x in np.linspace(slider2.start, slider2.end)])})
igplot2 = figure(height=400, width=400, title='Information Gain vs Threshold')
igplot2.line(x='x', y='y', source=igsource2)
igmaxlabel2 = Label(
    x=igsource2.data['x'][igsource2.data['y'].argmax()],
    y=igsource2.data['y'].max(),
    text=f"Max Info Gain: ({igsource2.data['x'][igsource2.data['y'].argmax()]:.3f}, {igsource2.data['y'].max():.3f})",
    text_font_size='10pt')
igplot2.add_layout(igmaxlabel2)

div2 = Div(text=f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider2.value, 1)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider2.value, 1)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider2.value, 1)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider2.value, 1)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider2.value, 1)[4]:.3f}</li>
    </ul>""")

def callback2(attr, old, new):
    source2.data['x'] = [slider2.value, slider2.value]
    div2.text = f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider2.value, 1)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider2.value, 1)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider2.value, 1)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider2.value, 1)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider2.value, 1)[4]:.3f}</li>
    </ul>"""

slider2.on_change('value', callback2)

plot2.line(x='x', y='y', source=source2, line_width=3)
plot_and_slider2 = row(plot2, column(widgetbox(slider2), div2, igplot2))

tabs.append(Panel(child=plot_and_slider2, title=h1_df.index[1]))

# third tab

slider3 = Slider(
    title='Threshold',
    start=min(pulsar_stars_df.iloc[:, 2]) - 0.1,
    end=max(pulsar_stars_df.iloc[:, 2]) + 0.1,
    step=(max(pulsar_stars_df.iloc[:, 2]) - min(pulsar_stars_df.iloc[:, 2]))/75,
    value=min(pulsar_stars_df.iloc[:, 2]) - 0.1
)

source3 = sources[2]

plot3 = figure()
plot3.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h0_quads[2]), color='red', legend='Not Star', fill_alpha=0.8)
plot3.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h1_quads[2]), color='green', legend='Star', fill_alpha=0.8)
plot3.legend.click_policy="hide"

igsource3 = ColumnDataSource(data={'x': np.linspace(slider3.start, slider3.end), 'y': np.array([shannon_entropy(x, 2)[4] for x in np.linspace(slider3.start, slider3.end)])})
igplot3 = figure(height=400, width=400, title='Information Gain vs Threshold')
igplot3.line(x='x', y='y', source=igsource3)
igmaxlabel3 = Label(
    x=igsource3.data['x'][igsource3.data['y'].argmax()],
    y=igsource3.data['y'].max(),
    text=f"Max Info Gain: ({igsource3.data['x'][igsource3.data['y'].argmax()]:.3f}, {igsource3.data['y'].max():.3f})",
    text_font_size='10pt')
igplot3.add_layout(igmaxlabel3)

div3 = Div(text=f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider3.value, 2)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider3.value, 2)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider3.value, 2)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider3.value, 2)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider3.value, 2)[4]:.3f}</li>
    </ul>""")

def callback3(attr, old, new):
    source3.data['x'] = [slider3.value, slider3.value]
    div3.text = f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider3.value, 2)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider3.value, 2)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider3.value, 2)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider3.value, 2)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider3.value, 2)[4]:.3f}</li>
    </ul>"""

slider3.on_change('value', callback3)

plot3.line(x='x', y='y', source=source3, line_width=3)
plot_and_slider3 = row(plot3, column(widgetbox(slider3), div3, igplot3))

tabs.append(Panel(child=plot_and_slider3, title=h1_df.index[2]))

# fourth tab

slider4 = Slider(
    title='Threshold',
    start=min(pulsar_stars_df.iloc[:, 3]) - 0.1,
    end=max(pulsar_stars_df.iloc[:, 3]) + 0.1,
    step=(max(pulsar_stars_df.iloc[:, 3]) - min(pulsar_stars_df.iloc[:, 3]))/75,
    value=min(pulsar_stars_df.iloc[:, 3]) - 0.1
)

source4 = sources[3]

plot4 = figure()
plot4.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h0_quads[3]), color='red', legend='Not Star', fill_alpha=0.8)
plot4.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h1_quads[3]), color='green', legend='Star', fill_alpha=0.8)
plot4.legend.click_policy="hide"

igsource4 = ColumnDataSource(data={'x': np.linspace(slider4.start, slider4.end), 'y': np.array([shannon_entropy(x, 3)[4] for x in np.linspace(slider4.start, slider4.end)])})
igplot4 = figure(height=400, width=400, title='Information Gain vs Threshold')
igplot4.line(x='x', y='y', source=igsource4)
igmaxlabel4 = Label(
    x=igsource4.data['x'][igsource4.data['y'].argmax()],
    y=igsource4.data['y'].max(),
    text=f"Max Info Gain: ({igsource4.data['x'][igsource4.data['y'].argmax()]:.3f}, {igsource4.data['y'].max():.3f})",
    text_font_size='10pt')
igplot4.add_layout(igmaxlabel4)

div4 = Div(text=f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider4.value, 3)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider4.value, 3)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider4.value, 3)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider4.value, 3)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider4.value, 3)[4]:.3f}</li>
    </ul>""")

def callback4(attr, old, new):
    source4.data['x'] = [slider4.value, slider4.value]
    div4.text = f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider4.value, 3)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider4.value, 3)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider4.value, 3)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider4.value, 3)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider4.value, 3)[4]:.3f}</li>
    </ul>"""

slider4.on_change('value', callback4)

plot4.line(x='x', y='y', source=source4, line_width=3)
plot_and_slider4 = row(plot4, column(widgetbox(slider4), div4, igplot4))

tabs.append(Panel(child=plot_and_slider4, title=h1_df.index[3]))

# fifth tab

slider5 = Slider(
    title='Threshold',
    start=min(pulsar_stars_df.iloc[:, 4]) - 0.1,
    end=max(pulsar_stars_df.iloc[:, 4]) + 0.1,
    step=(max(pulsar_stars_df.iloc[:, 4]) - min(pulsar_stars_df.iloc[:, 4]))/75,
    value=min(pulsar_stars_df.iloc[:, 4]) - 0.1
)

source5 = sources[4]

plot5 = figure()
plot5.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h0_quads[4]), color='red', legend='Not Star', fill_alpha=0.8)
plot5.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h1_quads[4]), color='green', legend='Star', fill_alpha=0.8)
plot5.legend.click_policy="hide"

igsource5 = ColumnDataSource(data={'x': np.linspace(slider5.start, slider5.end), 'y': np.array([shannon_entropy(x, 4)[4] for x in np.linspace(slider5.start, slider5.end)])})
igplot5 = figure(height=400, width=400, title='Information Gain vs Threshold')
igplot5.line(x='x', y='y', source=igsource5)
igmaxlabel5 = Label(
    x=igsource5.data['x'][igsource5.data['y'].argmax()],
    y=igsource5.data['y'].max(),
    text=f"Max Info Gain: ({igsource5.data['x'][igsource5.data['y'].argmax()]:.3f}, {igsource5.data['y'].max():.3f})",
    text_font_size='10pt')
igplot5.add_layout(igmaxlabel5)

div5 = Div(text=f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider5.value, 4)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider5.value, 4)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider5.value, 4)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider5.value, 4)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider5.value, 4)[4]:.3f}</li>
    </ul>""")

def callback5(attr, old, new):
    source5.data['x'] = [slider5.value, slider5.value]
    div5.text = f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider5.value, 4)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider5.value, 4)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider5.value, 4)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider5.value, 4)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider5.value, 4)[4]:.3f}</li>
    </ul>"""

slider5.on_change('value', callback5)

plot5.line(x='x', y='y', source=source5, line_width=3)
plot_and_slider5 = row(plot5, column(widgetbox(slider5), div5, igplot5))

tabs.append(Panel(child=plot_and_slider5, title=h1_df.index[4]))

# sixth tab

slider6 = Slider(
    title='Threshold',
    start=min(pulsar_stars_df.iloc[:, 5]) - 0.1,
    end=max(pulsar_stars_df.iloc[:, 5]) + 0.1,
    step=(max(pulsar_stars_df.iloc[:, 5]) - min(pulsar_stars_df.iloc[:, 5]))/75,
    value=min(pulsar_stars_df.iloc[:, 5]) - 0.1
)

source6 = sources[5]

plot6 = figure()
plot6.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h0_quads[5]), color='red', legend='Not Star', fill_alpha=0.8)
plot6.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h1_quads[5]), color='green', legend='Star', fill_alpha=0.8)
plot6.legend.click_policy="hide"

igsource6 = ColumnDataSource(data={'x': np.linspace(slider6.start, slider6.end), 'y': np.array([shannon_entropy(x, 5)[4] for x in np.linspace(slider6.start, slider6.end)])})
igplot6 = figure(height=400, width=400, title='Information Gain vs Threshold')
igplot6.line(x='x', y='y', source=igsource6)
igmaxlabel6 = Label(
    x=igsource6.data['x'][igsource6.data['y'].argmax()],
    y=igsource6.data['y'].max(),
    text=f"Max Info Gain: ({igsource6.data['x'][igsource6.data['y'].argmax()]:.3f}, {igsource6.data['y'].max():.3f})",
    text_font_size='10pt')
igplot6.add_layout(igmaxlabel6)

div6 = Div(text=f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider6.value, 5)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider6.value, 5)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider6.value, 5)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider6.value, 5)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider6.value, 5)[4]:.3f}</li>
    </ul>""")

def callback6(attr, old, new):
    source6.data['x'] = [slider6.value, slider6.value]
    div6.text = f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider6.value, 5)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider6.value, 5)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider6.value, 5)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider6.value, 5)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider6.value, 5)[4]:.3f}</li>
    </ul>"""

slider6.on_change('value', callback6)

plot6.line(x='x', y='y', source=source6, line_width=3)
plot_and_slider6 = row(plot6, column(widgetbox(slider6), div6, igplot6))

tabs.append(Panel(child=plot_and_slider6, title=h1_df.index[5]))

# seventh tab

slider7 = Slider(
    title='Threshold',
    start=min(pulsar_stars_df.iloc[:, 6]) - 0.1,
    end=max(pulsar_stars_df.iloc[:, 6]) + 0.1,
    step=(max(pulsar_stars_df.iloc[:, 6]) - min(pulsar_stars_df.iloc[:, 6]))/75,
    value=min(pulsar_stars_df.iloc[:, 6]) - 0.1
)

source7 = sources[6]

plot7 = figure()
plot7.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h0_quads[6]), color='red', legend='Not Star', fill_alpha=0.8)
plot7.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h1_quads[6]), color='green', legend='Star', fill_alpha=0.8)
plot7.legend.click_policy="hide"

igsource7 = ColumnDataSource(data={'x': np.linspace(slider7.start, slider7.end), 'y': np.array([shannon_entropy(x, 6)[4] for x in np.linspace(slider7.start, slider7.end)])})
igplot7 = figure(height=400, width=400, title='Information Gain vs Threshold')
igplot7.line(x='x', y='y', source=igsource7)
igmaxlabel7 = Label(
    x=igsource7.data['x'][igsource7.data['y'].argmax()],
    y=igsource7.data['y'].max(),
    text=f"Max Info Gain: ({igsource7.data['x'][igsource7.data['y'].argmax()]:.3f}, {igsource7.data['y'].max():.3f})",
    text_font_size='10pt')
igplot7.add_layout(igmaxlabel7)

div7 = Div(text=f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider7.value, 6)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider7.value, 6)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider7.value, 6)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider7.value, 6)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider7.value, 6)[4]:.3f}</li>
    </ul>""")

def callback7(attr, old, new):
    source7.data['x'] = [slider7.value, slider7.value]
    div7.text = f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider7.value, 6)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider7.value, 6)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider7.value, 6)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider7.value, 6)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider7.value, 6)[4]:.3f}</li>
    </ul>"""

slider7.on_change('value', callback7)

plot7.line(x='x', y='y', source=source7, line_width=3)
plot_and_slider7 = row(plot7, column(widgetbox(slider7), div7, igplot7))

tabs.append(Panel(child=plot_and_slider7, title=h1_df.index[6]))

# eighth tab

slider8 = Slider(
    title='Threshold',
    start=min(pulsar_stars_df.iloc[:, 7]) - 0.1,
    end=max(pulsar_stars_df.iloc[:, 7]) + 0.1,
    step=(max(pulsar_stars_df.iloc[:, 7]) - min(pulsar_stars_df.iloc[:, 7]))/75,
    value=min(pulsar_stars_df.iloc[:, 7]) - 0.1
)

source8 = sources[7]

plot8 = figure()
plot8.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h0_quads[7]), color='red', legend='Not Star', fill_alpha=0.8)
plot8.quad(left='left', right='right', top='top', bottom=0, source=ColumnDataSource(h1_quads[7]), color='green', legend='Star', fill_alpha=0.8)
plot8.legend.click_policy="hide"

igsource8 = ColumnDataSource(data={'x': np.linspace(slider8.start, slider8.end), 'y': np.array([shannon_entropy(x, 7)[4] for x in np.linspace(slider8.start, slider8.end)])})
igplot8 = figure(height=400, width=400, title='Information Gain vs Threshold')
igplot8.line(x='x', y='y', source=igsource8)
igmaxlabel8 = Label(
    x=igsource8.data['x'][igsource8.data['y'].argmax()],
    y=igsource8.data['y'].max(),
    text=f"Max Info Gain: ({igsource8.data['x'][igsource8.data['y'].argmax()]:.3f}, {igsource8.data['y'].max():.3f})",
    text_font_size='10pt')
igplot8.add_layout(igmaxlabel8)

div8 = Div(text=f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider8.value, 7)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider8.value, 7)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider8.value, 7)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider8.value, 7)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider8.value, 7)[4]:.3f}</li>
    </ul>""")

def callback8(attr, old, new):
    source8.data['x'] = [slider8.value, slider8.value]
    div8.text = f"""
    <ul>
        <li>Parent Ent: &nbsp;&nbsp;&nbsp;{shannon_entropy(slider8.value, 7)[0]:.3f}</li>
        <li>Left Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider8.value, 7)[1]:.3f}</li>
        <li>Right Ent: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider8.value, 7)[2]:.3f}</li>
        <li>Avg Wtd Ent: {shannon_entropy(slider8.value, 7)[3]:.3f}</li>
        <li>Info Gain: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{shannon_entropy(slider8.value, 7)[4]:.3f}</li>
    </ul>"""

slider8.on_change('value', callback8)

plot8.line(x='x', y='y', source=source8, line_width=3)
plot_and_slider8 = row(plot8, column(widgetbox(slider8), div8, igplot8))

tabs.append(Panel(child=plot_and_slider8, title=h1_df.index[7]))


# final
layout = Tabs(tabs=tabs)
curdoc().add_root(layout)

# bokeh serve entropy_slider.py