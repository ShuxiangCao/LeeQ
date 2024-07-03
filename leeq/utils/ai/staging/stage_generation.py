from typing import List

from .stage_execution import Stage


def stages_to_html(stages_list):
    stages_dict = {stage.label: stage.to_dict() for stage in stages_list}

    html_content = '<div style="font-family: Arial, sans-serif;">'

    # Loop through each stage in the dictionary
    for stage_key, stage_info in stages_dict.items():
        # Skip "Complete" and "Fail" stages
        if stage_key in ["Complete", "Fail"]:
            continue

        # Adding HTML content for each stage
        html_content += f'''
            <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
                <h3>{stage_info['Title']}</h3>
                <p><strong>Overview:</strong> {stage_info['Overview']}</p>
                <p><strong>Description:</strong> {stage_info['ExperimentDescription']}</p>
                <p><strong>Next Steps:</strong> {stage_info['Next']}</p>
            </div>
        '''

    html_content += '</div>'
    return html_content

#- Note: You can execute one experiment at each stage. If the stage should consider multiple runs with different argument of the same experiment,
#            implement it by suggesting the transitions rules that direct it itself.

def get_stages_from_description(description: str) -> List[Stage]:
    """
    Get stages from the description of the experiment.

    Parameters:
        description (str): The description of the experiment.

    Returns:
        List[Stage]: The list of stages of the experiment.
    """
    # Note: The same experiment with different parameter choice (very common when you need to refine the parameters) needs to be classified into the same stage. #

    prompt = """
**Objective**: Develop a systematic workflow for a series of scientific experiments involving sequential function calls to operate experimental equipment. The outcomes of each stage guide the next, ensuring logical progression. 

**Overview**: Summarize the requirement of the experiment. Only include the information from the input, do not add any additional information of your own.

**Instructions**:

- **Stages**: Divide the experiment into distinct stages, each representing a specific operation. 
Note: Generate as less stages as possible, ideally just one stage, but make sure each stage is distinct and has a clear purpose.
Note: If multiple sets of parameters are used for the same experiment, they should be considered into different stages.
Note: The data and result analysis and interpretation should not be considered as a stage.
Note: Refinement of the parameters should be included in the same stage, not in a separate stage.

- **Experiment Description**: rovide a detailed procedural outline for each stage of the experiment. The description
 should explicitly state the name of the experiment, list all parameters involved, and clearly outline the steps to be
 taken. This information will be distributed among various team members, who will carry out the tasks. Ensure that each
 instruction is clear and self-sufficient, enabling team members to execute their respective parts without needing
 additional context or clarification. Do not include objectives or goals in the description.

- **Stage Transitions**:
    - **Advance**: Define conditions for progressing to the next stage.
    - **Retry**: Specify when to repeat a stage with adjustments.
    - **Revert**: Return to the previous stage.
    
- **Output Format**: Present these instructions and conditions in a JSON format, with each stage as a key detailing the experiment and transition rules. 
The NEXT key must be a string detailing the transition conditions. Do not use "retry", "advance", or "revert", instead describe the stage label directly.

**Example**:

```json
{
    "Overview": <Summary of the experiment>
  "Stage1": {
    "Title": "Experiment1",
    "ExperimentDescription": "Conduct the <experiment name 1> with parameters <parameter list for experiment 1>.",
    "Next": "Proceed to Stage2 if successful, adjust the parameter based on the results suggestion and retry Stage1 if not. After 3 failures, proceed to Fail."
  },
  "Stage2": {
    "Title": "Experiment2",
    "ExperimentDescription": "Conduct the <experiment name 2> with parameters <parameter list for experiment 2>.",
    "Next": "Advance to Stage3 if standards are met, retry Stage2 with adjustments from results suggestions otherwise.After 3 failures, proceed to Fail."
  },
  "Stage3": {
    "Title": "Experiment3",
    "ExperimentDescription": "Conduct the <experiment name 3> with parameters <parameter list for experiment 3>.",
    "Next": "Move to Complete if successful, return to Stage3 if inconclusive. After 3 failures, proceed to Fail."
  },
  "Complete": {
    "Title": "Completion",
    "ExperimentDescription": "Conclude the experiment has succeeded.",
    "Next": "None"
  },
  "Fail": {
    "Title": "Failure",
    "ExperimentDescription": "Conclude the experiment has failed.",
    "Next": "None"
  }
}
```
**Deliverables**: Provide a complete JSON detailing all stages, descriptions, and transition criteria for efficient experimentation and analysis. Only return the JSON object.
    """

    completed_prompt = prompt + description
    import mllm
    chat = mllm.Chat(completed_prompt, "You are a very smart and helpful assistant who only reply in JSON dict")
    res = chat.complete(parse="dict", expensive=True, cache=True)
    stages = []

    overview = res.pop("Overview")

    for stage_name, stage_content in res.items():
        stage = Stage(label=stage_name, title=stage_content['Title'],
                      overview=overview,
                      description=stage_content['ExperimentDescription'],
                      next_stage_guide=stage_content['Next'])
        stages.append(stage)

    return stages


if __name__ == '__main__':
    description_ramsey = '''
## Thought

Ramsey experiment can predict the qubit frequency different to the frequency I am driving it. First I guess a qubit frequency (which already set in the system), and assume the difference is no more than 10 MHz. Therefore I run a Ramsey experiment with frequency offset 10 MHz. Then I wish to do a more accurate calibration by increase the experiment time, and reduce the offset to 1MHz. If this experiment failed and show a value more than 3 MHz its likely that the initial guess is more than 10MHz away from the qubit. Therefore we go back and run experiment at 20MHz offset again. After it succeeded, we do a fine calibration with offset 0.1MHz.

## Steps

- Run Ramsey experiment on the qubit, with frequency offset 10 MHz, stop at 0.3us, step 0.005us.
- Extract the number of periods from the AI analysis texts.
- If observed less than 3 period, double the stop value and step value and try again.
- If observed more than 10 period, half the stop value and step value and try again.
- Run Ramsey experiment on the qubit, with frequency offset 1 MHz, stop at 3us, step 0.05us
- If the second experiment obtained a frequency offset more than 3MHz, go back to the first step, set frequency offset to 20MHz. and try again.
- Run Ramsey experiment on the qubit, with frequency offset 0.1 MHz, stop at 30us, step 0.5us.
'''
    description_rabi = '''

    # Gate Amplitude Calibration

## Thought

To accurately calibrate the amplitude of the control pulses for our qubit gates, we start with a Rabi oscillation experiment. This experiment helps determine the amplitude required to perform a full rotation on the Bloch sphere. We begin the calibration with a preliminary range of pulse durations starting from 0.01 microseconds up to 0.15 microseconds, incrementing by 0.001 microseconds each step. Successful determination of the Rabi frequency from these measurements will indicate the optimal amplitude setting for the qubit gates.

After successfully calibrating the Rabi frequency, we proceed to Pingpong amplitude calibration using the default parameters. This secondary calibration further refines our control over the qubit by adjusting the amplitudes based on the results from the Rabi experiment, ensuring more precise and reliable gate operations.

## Steps

- Conduct a Rabi experiment to determine the Rabi frequency: Start pulse duration at 0.01 microseconds, step 0.001 microseconds, stop at 0.15 microseconds.
- If observed less than 3 period, double the stop value and step value and try again..
- If observed more than 10 period, half the stop value and step value and try again.
- If the above experiment failed, re-do it and adjust parameters based on visual instructions.
- Upon the successful completion of the Rabi experiment, run Pingpong amplitude calibration with default parameters.
'''

    description = description_rabi
    stages = get_stages_from_description(description)
    print(stages)
