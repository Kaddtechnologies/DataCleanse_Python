"""
AI Confidence Scoring
-------------------
This module contains functions for using AI (specifically OpenAI's API) to score
the confidence of potential duplicate matches.
"""

import os
import json
import asyncio
import aiohttp
import requests
from typing import List, Dict

async def apply_ai_confidence_scoring_async(master_records: List[Dict]) -> List[Dict]:
    """
    Apply AI-based confidence scoring to the master records asynchronously.
    
    Uses OpenAI's API to evaluate the confidence of each duplicate match.
    Processes all records and allows AI to update confidence levels based on its judgment.
    
    Args:
        master_records: List of master records with potential duplicates
        
    Returns:
        List[Dict]: The updated master records with AI confidence scores
    """
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OpenAI API key not found. Skipping AI confidence scoring.")
        return master_records
    
    try:
        # Process all records, but prioritize those with uncertain confidence
        # Uncertain confidence: overall score between 70-90
        records_for_ai = []
        record_indices = []
        
        for i, master in enumerate(master_records):
            # Check if any duplicates have uncertain confidence scores
            uncertain_dups = [
                dup for dup in master['Duplicates']
                if 70 <= dup['Overall_score'] <= 90
            ]
            
            if uncertain_dups:
                records_for_ai.append(master)
                record_indices.append(i)
        
        if not records_for_ai:
            print("No records found that need AI scoring.")
            return master_records
            
        print(f"Found {len(records_for_ai)} master records with duplicates for AI scoring.")
        
        # Process in smaller batches to avoid exceeding token limits
        batch_size = 3  # Process 3 master records at a time
        
        # Create batches
        batches = []
        for batch_idx in range(0, len(records_for_ai), batch_size):
            batch = records_for_ai[batch_idx:batch_idx+batch_size]
            batches.append((batch_idx, batch))
        
        # Process batches asynchronously
        async def process_batch(batch_idx, batch):
            # Prepare data for the AI
            entries = []
            for master in batch:
                # Add master record
                master_entry = f"Master: {master['MasterName']} | {master['MasterAddress']}"
                entries.append(master_entry)
                
                # Add all duplicates, with focus on uncertain ones
                for dup in master['Duplicates']:
                    confidence_level = "uncertain" if 70 <= dup['Overall_score'] <= 90 else "high" if dup['Overall_score'] > 90 else "low"
                    dup_entry = f"Duplicate: {dup['Name']} | {dup['Address']} | Score: {dup['Overall_score']} | Confidence: {confidence_level}"
                    entries.append(dup_entry)
            
            # Create prompt for OpenAI - focused on evaluating and updating confidence
            prompt = """
            You are an AI data deduplication and scoring assistant with expertise in evaluating duplicate records.
            
            Given a list of data entries with potential duplicates, your task is to:
            1. For each duplicate entry, evaluate how likely it is a true duplicate of the master record.
            2. Assign a confidence score (between 0 and 1) that represents your assessment.
            3. Pay special attention to entries marked as "uncertain" confidence, but evaluate all entries.
            4. Consider name similarity, address similarity, and any other relevant factors.
            5. If you believe the existing score is incorrect (too high or too low), your score should reflect your best judgment.
            
            Guidelines for confidence scoring:
            - 0.9-1.0: Definite duplicate (nearly identical records)
            - 0.8-0.9: Very likely duplicate (minor variations but clearly the same entity)
            - 0.7-0.8: Probable duplicate (some differences but likely the same entity)
            - 0.5-0.7: Possible duplicate (significant differences, but could be the same entity)
            - 0.0-0.5: Unlikely to be a duplicate (major differences, likely different entities)
            
            Process the following data entries:
            
            {}
            
            Please output a JSON array with objects like:
            [
              {{
                "entry": "<duplicate entry>",
                "confidence": <float between 0 and 1>,
                "reasoning": "<brief explanation for your confidence score>"
              }},
              ...
            ]
            """.format("\n".join(entries))
            
            print(f"Processing batch {batch_idx//batch_size + 1}/{len(batches)}...")
            
            # Call OpenAI API with a timeout
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "gpt-4.1-nano",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.2
                        },
                        timeout=aiohttp.ClientTimeout(total=30)  # 30 second timeout
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            ai_content = result["choices"][0]["message"]["content"]
                            
                            try:
                                # Parse AI response
                                ai_scores = json.loads(ai_content)
                                
                                # Update confidence scores in the filtered master records
                                score_index = 0
                                for master_idx, master in enumerate(batch):
                                    for dup_idx, dup in enumerate(master['Duplicates']):
                                        if score_index < len(ai_scores):
                                            # Add AI confidence score
                                            # Update the original master_records list using the stored indices
                                            original_idx = record_indices[batch_idx + master_idx]
                                            ai_conf = ai_scores[score_index]['confidence']
                                            reasoning = ai_scores[score_index].get('reasoning', 'No reasoning provided')
                                            
                                            # Update the confidence score
                                            master_records[original_idx]['Duplicates'][dup_idx]['LLM_conf'] = ai_conf
                                            master_records[original_idx]['Duplicates'][dup_idx]['LLM_reasoning'] = reasoning
                                            
                                            # Log the AI confidence score
                                            print(f"AI confidence for {dup['Name']}: {ai_conf} - {reasoning[:50]}...")
                                            
                                            score_index += 1
                                
                                return True
                            except Exception as e:
                                print(f"Error parsing AI response: {str(e)}")
                                return False
                        else:
                            response_text = await response.text()
                            print(f"Error calling OpenAI API: {response.status} - {response_text}")
                            return False
            except asyncio.TimeoutError:
                print(f"Timeout when calling OpenAI API for batch {batch_idx//batch_size + 1}. Skipping this batch.")
                return False
            except Exception as e:
                print(f"Error in API call: {str(e)}")
                return False
        
        # Process all batches concurrently
        tasks = []
        for batch_idx, batch in batches:
            task = asyncio.create_task(process_batch(batch_idx, batch))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful batches
        successful_batches = sum(1 for result in results if result is True)
        print(f"AI confidence scoring completed for {successful_batches}/{len(batches)} batches.")
    
    except Exception as e:
        print(f"Error in AI confidence scoring: {str(e)}")
    
    return master_records

def apply_ai_confidence_scoring(master_records: List[Dict]) -> List[Dict]:
    """
    Synchronous wrapper for the asynchronous AI confidence scoring function.
    
    This function attempts to use the existing event loop if one is running,
    otherwise creates a new one. If that fails, it falls back to synchronous processing.
    
    Args:
        master_records: List of master records with potential duplicates
        
    Returns:
        List[Dict]: The updated master records with AI confidence scores
    """
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print("Warning: Event loop is already running. Using synchronous processing instead.")
            # Fall back to synchronous processing
            return process_records_synchronously(master_records)
        else:
            # Use the existing loop
            return loop.run_until_complete(apply_ai_confidence_scoring_async(master_records))
    except RuntimeError:
        # No event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(apply_ai_confidence_scoring_async(master_records))
        finally:
            loop.close()

def process_records_synchronously(master_records: List[Dict]) -> List[Dict]:
    """
    Process records synchronously as a fallback when asyncio can't be used.
    
    This function performs the same AI confidence scoring as the async version,
    but does it sequentially instead of concurrently.
    
    Args:
        master_records: List of master records with potential duplicates
        
    Returns:
        List[Dict]: The updated master records with AI confidence scores
    """
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OpenAI API key not found. Skipping AI confidence scoring.")
        return master_records
    
    try:
        # Process all records, but prioritize those with uncertain confidence
        # Uncertain confidence: overall score between 70-90
        records_for_ai = []
        record_indices = []
        
        for i, master in enumerate(master_records):
            # Check if any duplicates have uncertain confidence scores
            uncertain_dups = [
                dup for dup in master['Duplicates']
                if 70 <= dup['Overall_score'] <= 90
            ]
            
            if uncertain_dups:
                records_for_ai.append(master)
                record_indices.append(i)
        
        if not records_for_ai:
            print("No records found that need AI scoring.")
            return master_records
            
        print(f"Found {len(records_for_ai)} master records with duplicates for AI scoring (synchronous mode).")
        
        # Process in smaller batches to avoid exceeding token limits
        batch_size = 3  # Process 3 master records at a time
        for batch_idx in range(0, len(records_for_ai), batch_size):
            batch = records_for_ai[batch_idx:batch_idx+batch_size]
            
            # Prepare data for the AI
            entries = []
            for master in batch:
                # Add master record
                master_entry = f"Master: {master['MasterName']} | {master['MasterAddress']}"
                entries.append(master_entry)
                
                # Add all duplicates, with focus on uncertain ones
                for dup in master['Duplicates']:
                    confidence_level = "uncertain" if 70 <= dup['Overall_score'] <= 90 else "high" if dup['Overall_score'] > 90 else "low"
                    dup_entry = f"Duplicate: {dup['Name']} | {dup['Address']} | Score: {dup['Overall_score']} | Confidence: {confidence_level}"
                    entries.append(dup_entry)
            
            # Create prompt for OpenAI - focused on evaluating and updating confidence
            prompt = """
            You are an AI data deduplication and scoring assistant with expertise in evaluating duplicate records.
            
            Given a list of data entries with potential duplicates, your task is to:
            1. For each duplicate entry, evaluate how likely it is a true duplicate of the master record.
            2. Assign a confidence score (between 0 and 1) that represents your assessment.
            3. Pay special attention to entries marked as "uncertain" confidence, but evaluate all entries.
            4. Consider name similarity, address similarity, and any other relevant factors.
            5. If you believe the existing score is incorrect (too high or too low), your score should reflect your best judgment.
            
            Guidelines for confidence scoring:
            - 0.9-1.0: Definite duplicate (nearly identical records)
            - 0.8-0.9: Very likely duplicate (minor variations but clearly the same entity)
            - 0.7-0.8: Probable duplicate (some differences but likely the same entity)
            - 0.5-0.7: Possible duplicate (significant differences, but could be the same entity)
            - 0.0-0.5: Unlikely to be a duplicate (major differences, likely different entities)
            
            Process the following data entries:
            
            {}
            
            Please output a JSON array with objects like:
            [
              {{
                "entry": "<duplicate entry>",
                "confidence": <float between 0 and 1>,
                "reasoning": "<brief explanation for your confidence score>"
              }},
              ...
            ]
            """.format("\n".join(entries))
            
            print(f"Processing batch {batch_idx//batch_size + 1}/{(len(records_for_ai) + batch_size - 1)//batch_size}...")
            
            # Call OpenAI API with a timeout
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4.1-nano",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2
                    },
                    timeout=30  # 30 second timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ai_content = result["choices"][0]["message"]["content"]
                    
                    try:
                        # Parse AI response
                        ai_scores = json.loads(ai_content)
                        
                        # Update confidence scores in the filtered master records
                        score_index = 0
                        for master_idx, master in enumerate(batch):
                            for dup_idx, dup in enumerate(master['Duplicates']):
                                if score_index < len(ai_scores):
                                    # Add AI confidence score
                                    # Update the original master_records list using the stored indices
                                    original_idx = record_indices[batch_idx + master_idx]
                                    ai_conf = ai_scores[score_index]['confidence']
                                    reasoning = ai_scores[score_index].get('reasoning', 'No reasoning provided')
                                    
                                    # Update the confidence score
                                    master_records[original_idx]['Duplicates'][dup_idx]['LLM_conf'] = ai_conf
                                    master_records[original_idx]['Duplicates'][dup_idx]['LLM_reasoning'] = reasoning
                                    
                                    # Log the AI confidence score
                                    print(f"AI confidence for {dup['Name']}: {ai_conf} - {reasoning[:50]}...")
                                    
                                    score_index += 1
                    except Exception as e:
                        print(f"Error parsing AI response: {str(e)}")
                else:
                    print(f"Error calling OpenAI API: {response.status_code} - {response.text}")
            except requests.exceptions.Timeout:
                print(f"Timeout when calling OpenAI API for batch {batch_idx//batch_size + 1}. Skipping this batch.")
            except Exception as e:
                print(f"Error in API call: {str(e)}")
    
    except Exception as e:
        print(f"Error in AI confidence scoring: {str(e)}")
    
    print(f"AI confidence scoring completed for {len(records_for_ai)} master records.")
    return master_records 