# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""PyMIALSRTK utils functions for resource monitoring originated from Nipype.

See https://nipype.readthedocs.io/en/latest/api/generated/nipype.utils.profiler.html for Nipype's documentation.
See https://nipype.readthedocs.io/en/latest/api/generated/nipype.utils.draw_gantt_chart.html
"""
# Import packages
import datetime
import simplejson as json

from nipype.utils.draw_gantt_chart import calculate_resource_timeseries,\
    create_event_dict, draw_nodes, draw_lines, draw_resource_bar

# Pandas
try:
    import pandas as pd
except ImportError:
    print(
        "Pandas not found; in order for full functionality of this module "
        "install the pandas package"
    )
    pass


# Log node stats function
def log_nodes_cb(node, status):
    """Function to record node run statistics to a log file as json
    dictionaries (source: https://github.com/nipy/nipype/blob/10388b0bb5d341628b68258f655b9f86fb2cf5a6/nipype/utils/profiler.py#L111-L153)

    Modified to ignore mapnodes.

    Parameters
    ----------
    node : nipype.pipeline.engine.Node
        the node being logged

    status : string
        acceptable values are 'start', 'end'; otherwise it is
        considered and error

    Returns
    -------
    None
        this function does not return any values, it logs the node
        status info to the callback logger

    """
    if status != "end":
        return

    # Ignore MapNodes by checking for lists, since
    # the subnodes already called
    # source: https://github.com/nipy/nipype/issues/2571#issuecomment-396209516
    if isinstance(node.result.runtime, list):
        return

    # Import packages
    import logging
    import json

    status_dict = {
        "name": node.name,
        "id": node._id,
        "start": getattr(node.result.runtime, "startTime"),
        "finish": getattr(node.result.runtime, "endTime"),
        "duration": getattr(node.result.runtime, "duration"),
        "runtime_threads": getattr(node.result.runtime, "cpu_percent", "N/A"),
        "runtime_memory_gb": getattr(node.result.runtime, "mem_peak_gb", "N/A"),
        "estimated_memory_gb": node.mem_gb,
        "num_threads": node.n_procs,
    }

    if status_dict["start"] is None or status_dict["finish"] is None:
        status_dict["error"] = True

    # Dump string to log
    logging.getLogger("callback").debug(json.dumps(status_dict))


def log_to_dict(logfile):
    """
    Function to extract log node dictionaries into a list of python
    dictionaries and return the list as well as the final node

    Parameters
    ----------
    logfile : string
        path to the json-formatted log file generated from a nipype
        workflow execution

    Returns
    -------
    nodes_list : list
        a list of python dictionaries containing the runtime info
        for each nipype node

    """
    # Init variables
    with open(logfile, "r") as content:
        # read file separating each line
        lines = content.readlines()

    nodes_list = []
    for l in lines:
        node = json.loads(l)
        node["start"] = datetime.datetime.strptime(node["start"],
                                                   '%Y-%m-%dT%H:%M:%S.%fZ')
        node["finish"] = datetime.datetime.strptime(node["finish"],
                                                    '%Y-%m-%dT%H:%M:%S.%fZ')
        nodes_list.append(node)

    # Return list of nodes
    return nodes_list


def generate_gantt_chart(
    logfile,
    cores,
    minute_scale=10,
    space_between_minutes=50,
    colors=["#7070FF", "#4E4EB2", "#2D2D66", "#9B9BFF"],
):
    """
    Generates a gantt chart in html showing the workflow execution based on a callback log file.
    This script was intended to be used with the MultiprocPlugin.
    The following code shows how to set up the workflow in order to generate the log file:

    Parameters
    ----------
    logfile : string
        filepath to the callback log file to plot the gantt chart of

    cores : integer
        the number of cores given to the workflow via the 'n_procs'
        plugin arg

    minute_scale : integer (optional); default=10
        the scale, in minutes, at which to plot line markers for the
        gantt chart; for example, minute_scale=10 means there are lines
        drawn at every 10 minute interval from start to finish

    space_between_minutes : integer (optional); default=50
        scale factor in pixel spacing between minute line markers

    colors : list (optional)
        a list of colors to choose from when coloring the nodes in the
        gantt chart

    Returns
    -------
    None
        the function does not return any value but writes out an html
        file in the same directory as the callback log path passed in

    Usage
    -----
    # import logging
    # import logging.handlers
    # from nipype.utils.profiler import log_nodes_cb
    # log_filename = 'callback.log'
    # logger = logging.getLogger('callback')
    # logger.setLevel(logging.DEBUG)
    # handler = logging.FileHandler(log_filename)
    # logger.addHandler(handler)
    # #create workflow
    # workflow = ...
    # workflow.run(plugin='MultiProc',
    #     plugin_args={'n_procs':8, 'memory':12, 'status_callback': log_nodes_cb})
    # generate_gantt_chart('callback.log', 8)

    """
    # add the html header
    html_string = """<!DOCTYPE html>
    <head>
        <style>
            #content{
                width:99%;
                height:100%;
                position:absolute;
            }
            .node{
                background-color:#7070FF;
                border-radius: 5px;
                position:absolute;
                width:20px;
                white-space:pre-wrap;
            }
            .line{
                position: absolute;
                color: #C2C2C2;
                opacity: 0.5;
                margin: 0px;
            }
            .time{
                position: absolute;
                font-size: 16px;
                color: #666666;
                margin: 0px;
            }
            .bar{
                position: absolute;
                height: 1px;
                opacity: 0.7;
            }
            .dot{
                position: absolute;
                width: 1px;
                height: 1px;
                background-color: red;
            }
            .label {
                width:20px;
                height:20px;
                opacity: 0.7;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <div id="content">
            <div style="display:inline-block;">
    """

    close_header = """
    </div>
    <div style="display:inline-block;margin-left:60px;vertical-align: top;">
        <p><span><div class="label" style="background-color:#90BBD7;"></div> Estimated Resource</span></p>
        <p><span><div class="label" style="background-color:#03969D;"></div> Actual Resource</span></p>
        <p><span><div class="label" style="background-color:#f00;"></div> Failed Node</span></p>
    </div>
    """

    # Read in json-log to get list of node dicts
    nodes_list = log_to_dict(logfile)

    # Create the header of the report with useful information
    start_node = nodes_list[0]
    last_node = nodes_list[-1]

    print(f'last_node["finish"]: {last_node["finish"]}')
    print(f'start_node["start"] :{start_node["start"]}')

    duration = (last_node["finish"] - start_node["start"]).total_seconds()

    # Get events based dictionary of node run stats
    events = create_event_dict(start_node["start"], nodes_list)

    # Summary strings of workflow at top
    html_string += (
        "<p>Start: " + start_node["start"].strftime("%Y-%m-%d %H:%M:%S") + "</p>"
    )
    html_string += (
        "<p>Finish: " + last_node["finish"].strftime("%Y-%m-%d %H:%M:%S") + "</p>"
    )
    html_string += "<p>Duration: " + "{0:.2f}".format(duration / 60) + " minutes</p>"
    html_string += "<p>Nodes: " + str(len(nodes_list)) + "</p>"
    html_string += "<p>Cores: " + str(cores) + "</p>"
    html_string += close_header
    # Draw nipype nodes Gantt chart and runtimes
    html_string += draw_lines(
        start_node["start"], duration, minute_scale, space_between_minutes
    )
    html_string += draw_nodes(
        start_node["start"],
        nodes_list,
        cores,
        minute_scale,
        space_between_minutes,
        colors,
    )

    # Get memory timeseries
    estimated_mem_ts = calculate_resource_timeseries(events, "estimated_memory_gb")
    runtime_mem_ts = calculate_resource_timeseries(events, "runtime_memory_gb")
    # Plot gantt chart
    resource_offset = 120 + 30 * cores
    html_string += draw_resource_bar(
        start_node["start"],
        last_node["finish"],
        estimated_mem_ts,
        space_between_minutes,
        minute_scale,
        "#90BBD7",
        resource_offset * 2 + 120,
        "Memory",
    )
    html_string += draw_resource_bar(
        start_node["start"],
        last_node["finish"],
        runtime_mem_ts,
        space_between_minutes,
        minute_scale,
        "#03969D",
        resource_offset * 2 + 120,
        "Memory",
    )

    # Get threads timeseries
    estimated_threads_ts = calculate_resource_timeseries(events, "estimated_threads")
    runtime_threads_ts = calculate_resource_timeseries(events, "runtime_threads")
    # Plot gantt chart
    html_string += draw_resource_bar(
        start_node["start"],
        last_node["finish"],
        estimated_threads_ts,
        space_between_minutes,
        minute_scale,
        "#90BBD7",
        resource_offset,
        "Threads",
    )
    html_string += draw_resource_bar(
        start_node["start"],
        last_node["finish"],
        runtime_threads_ts,
        space_between_minutes,
        minute_scale,
        "#03969D",
        resource_offset,
        "Threads",
    )

    # finish html
    html_string += """
        </div>
    </body>"""

    # save file
    with open(logfile + ".html", "w") as html_file:
        html_file.write(html_string)